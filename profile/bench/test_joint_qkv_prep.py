"""Correctness tests for lingbotvla/ops/joint_qkv_prep.py.

    python profile/bench/test_joint_qkv_prep.py            # CPU + GPU (if available)
    TRITON_INTERPRET=1 python profile/bench/test_joint_qkv_prep.py --interpret                   # Triton kernels on CPU (interpreter)
    JOINT_QKV_PREP_BLOCK_T=32 ... python profile/bench/test_joint_qkv_prep.py                   # try another tile

Checks, against the current eager semantics (joint_qkv_prep_reference = cat + HF apply_rotary_pos_emb + transpose):
  1. forward Q/K/V equal (exact on CPU and GPU: the kernels are compiled with enable_fp_fusion=False)
  2. backward grads of the 6 inputs equal (same tolerances), VLM grads come back in the VLM dtype
  3. fp64 gradcheck of the autograd.Function on a tiny shape (CPU)
  4. GPU only: flex_attention on the un-padded [B, H, 337, D] output equals flex on the padded-to-384 reference
     (forward and grads), i.e. dropping the pad is safe on this torch version; eagerly (bf16 grads within one bf16
     rounding step) and compiled through the repo's flex helpers at the training shape (forward, dQ/dK/dV)
  5. GPU only: timing of the fused forward/backward vs the eager chain at the training shape
  6. NaN/Inf in the upstream gradients come out where eager puts them (CPU, interpreter, GPU)
Prints PASS/FAIL per check and exits non-zero on failure.
"""
import os
import sys
import time

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, _REPO)
try:
    import lingbotvla.ops  # noqa: F401  (pulls transformers via lingbotvla/ops/__init__.py)
except ImportError:
    # machine without transformers (CPU-only test box): expose lingbotvla.ops as a bare namespace package
    import types
    for _name, _sub in (("lingbotvla", "lingbotvla"), ("lingbotvla.ops", "lingbotvla/ops")):
        _m = types.ModuleType(_name)
        _m.__path__ = [os.path.join(_REPO, _sub)]
        sys.modules[_name] = _m
from lingbotvla.ops.joint_qkv_prep import joint_qkv_prep, joint_qkv_prep_reference  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILED.append(name)


# fp32 -> bf16 rounding helper of the kernels, on bit patterns. NaN payloads whose low half makes the +0x7FFF carry
# overflow (CUDA's canonical NaN 0x7FFFFFFF among them; float('nan') = 0x7FC00000 is not one of them) must stay NaN;
# everything else must equal torch's .to(bfloat16).
PAYLOADS = [0x7FFFFFFF, 0xFFFFFFFF, 0x7F800001, 0x7F808000, 0xFF800001, 0x7FC00000, 0xFFC00000,
            0x7F800000, 0xFF800000, 0x7F7FFFFF, 0x00000001, 0x80000000, 0x3F800000, 0x3F808000, 0x3F818000]
try:
    import triton
    import triton.language as tl
    from lingbotvla.ops.joint_qkv_prep import _f32_to_bf16_bits_rtne as _rtne_helper  # noqa: E402
except Exception:  # pragma: no cover
    triton = None

if triton is not None:

    @triton.jit
    def _rtne_bits_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
        i = tl.arange(0, BLOCK)
        m = i < N
        tl.store(y_ptr + i, _rtne_helper(tl.load(x_ptr + i, mask=m, other=0.0)), mask=m)


def run_rounding_payloads(device, tag):
    bits = torch.tensor([p - (1 << 32) if p >= 1 << 31 else p for p in PAYLOADS], dtype=torch.int32)
    x = bits.view(torch.float32).to(device)
    y = torch.empty(len(PAYLOADS), dtype=torch.int16, device=device)
    _rtne_bits_kernel[(1,)](x, y, len(PAYLOADS), BLOCK=16)
    y = y.cpu().int() & 0xFFFF
    ref = x.cpu().to(torch.bfloat16).view(torch.int16).int() & 0xFFFF
    nan_in = torch.isnan(x.cpu())
    y_nan = ((y & 0x7F80) == 0x7F80) & ((y & 0x7F) != 0)
    ok = torch.equal(y_nan, nan_in) and torch.equal(y[~nan_in], ref[~nan_in])
    check(f"{tag} fp32->bf16 rounding on NaN payloads and edge cases", ok,
          " ".join(f"{p:08x}->{int(v):04x}" for p, v in zip(PAYLOADS, y)))


def make_inputs(B, Lv, Le, Hq, Hkv, D, device, vlm_dtype=torch.bfloat16, exp_dtype=torch.float32, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = lambda *s, dt: torch.randn(*s, generator=g).to(device=device, dtype=dt).requires_grad_(True)  # noqa: E731
    q_v, k_v, v_v = r(B, Lv, Hq, D, dt=vlm_dtype), r(B, Lv, Hkv, D, dt=vlm_dtype), r(B, Lv, Hkv, D, dt=vlm_dtype)
    q_e, k_e, v_e = r(B, Le, Hq, D, dt=exp_dtype), r(B, Le, Hkv, D, dt=exp_dtype), r(B, Le, Hkv, D, dt=exp_dtype)
    L = Lv + Le
    pos = torch.arange(L, dtype=torch.float32).view(1, L, 1).expand(B, L, D // 2) * (0.37 + torch.arange(D // 2) / D)
    emb = torch.cat([pos, pos], dim=-1)  # both halves identical, like HF emb = cat(freqs, freqs)
    cos, sin = emb.cos().to(device), emb.sin().to(device)
    return (q_v, k_v, v_v, q_e, k_e, v_e), cos, sin


def fwd_bwd(fn, inputs, cos, sin, upstream):
    outs = fn(*inputs, cos, sin)
    loss = sum((o.float() * u).sum() for o, u in zip(outs, upstream))
    grads = torch.autograd.grad(loss, inputs)
    return [o.detach() for o in outs], [g.detach() for g in grads]


def compare(name, a, b, rtol, atol):
    ok = len(a) == len(b) and all(x.shape == y.shape and x.dtype == y.dtype for x, y in zip(a, b))
    worst = 0.0
    for x, y in zip(a, b):
        x, y = x.float(), y.float()
        if not torch.allclose(x, y, rtol=rtol, atol=atol):
            ok = False
        worst = max(worst, (x - y).abs().max().item())
    check(name, ok, f"max|diff|={worst:.3e}")


def run_equivalence(device, B, Lv, Le, Hq, Hkv, D, rtol, atol, tag):
    inputs, cos, sin = make_inputs(B, Lv, Le, Hq, Hkv, D, device)
    gen = torch.Generator(device="cpu").manual_seed(1)
    L = Lv + Le
    upstream = [torch.randn(B, h, L, D, generator=gen).to(device) for h in (Hq, Hkv, Hkv)]
    ref_o, ref_g = fwd_bwd(joint_qkv_prep_reference, inputs, cos, sin, upstream)
    new_o, new_g = fwd_bwd(joint_qkv_prep, inputs, cos, sin, upstream)
    compare(f"{tag} forward Q/K/V", new_o, ref_o, rtol, atol)
    compare(f"{tag} backward grads", new_g, ref_g, rtol, atol)
    check(f"{tag} grad dtypes", all(g.dtype == i.dtype for g, i in zip(new_g, inputs)),
          str([str(g.dtype).replace("torch.", "") for g in new_g]))
    check(f"{tag} output contiguous", all(o.is_contiguous() for o in new_o))


def run_nonfinite(device, tag):
    """NaN/Inf in the upstream gradients must come out where eager puts them, the bf16 VLM grads included."""
    B, Lv, Le, Hq, Hkv, D = 2, 13, 5, 4, 2, 16
    inputs, cos, sin = make_inputs(B, Lv, Le, Hq, Hkv, D, device)
    L = Lv + Le
    gen = torch.Generator(device="cpu").manual_seed(2)
    upstream = [torch.randn(B, h, L, D, generator=gen) for h in (Hq, Hkv, Hkv)]
    for u in upstream:
        u[0, 0, 3, 1] = float("nan")  # VLM rows
        u[1, 1, 2, 9] = float("inf")
        u[0, 1, Lv + 1, 4] = float("nan")  # expert rows
        u[1, 0, Lv + 2, 12] = -float("inf")
    upstream = [u.to(device) for u in upstream]
    _, ref_g = fwd_bwd(joint_qkv_prep_reference, inputs, cos, sin, upstream)
    _, new_g = fwd_bwd(joint_qkv_prep, inputs, cos, sin, upstream)
    ok, n = True, 0
    for a, b in zip(new_g, ref_g):
        a, b = a.float(), b.float()
        nan, inf = torch.isnan(b), torch.isinf(b)
        fin = ~(nan | inf)
        ok = ok and torch.equal(torch.isnan(a), nan) and torch.equal(a[inf], b[inf]) and torch.equal(a[fin], b[fin])
        n += int(nan.sum()) + int(inf.sum())
    check(f"{tag} non-finite upstream grads", ok, f"({n} NaN/Inf grad elements, all where eager has them)")


def run_bf16_out(device, B, Lv, Le, Hq, Hkv, D, rtol, atol, tag):
    """out_dtype=bf16 (--train.attention_dtype=bf16): forward == reference fp32 result rounded RTNE to bf16;
    backward with bf16 upstream grads == reference grads (the cast's gradient is the identity)."""
    inputs, cos, sin = make_inputs(B, Lv, Le, Hq, Hkv, D, device, seed=9)
    gen = torch.Generator(device="cpu").manual_seed(2)
    L = Lv + Le
    upstream = [torch.randn(B, h, L, D, generator=gen).to(device).to(torch.bfloat16) for h in (Hq, Hkv, Hkv)]
    ref_o, ref_g = fwd_bwd(lambda *a: [x.to(torch.bfloat16) for x in joint_qkv_prep_reference(*a)], inputs, cos, sin, upstream)
    new_o, new_g = fwd_bwd(lambda *a: joint_qkv_prep(*a, out_dtype=torch.bfloat16), inputs, cos, sin, upstream)
    check(f"{tag} bf16 out dtypes", all(o.dtype == torch.bfloat16 for o in new_o))
    compare(f"{tag} bf16 out forward Q/K/V", new_o, ref_o, rtol, atol)
    compare(f"{tag} bf16 out backward grads", new_g, ref_g, rtol, atol)


def run_gradcheck():
    B, Lv, Le, Hq, Hkv, D = 2, 5, 3, 2, 1, 8
    inputs, cos, sin = make_inputs(B, Lv, Le, Hq, Hkv, D, "cpu", torch.float64, torch.float64)
    cos, sin = cos.double(), sin.double()
    fn = lambda *xs: joint_qkv_prep(*xs, cos, sin, torch.float64)  # noqa: E731
    try:
        ok = torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-6, rtol=1e-5)
    except Exception as e:  # pragma: no cover
        ok = False
        print("  gradcheck error:", str(e)[:300])
    check("fp64 gradcheck (CPU)", ok)


def run_flex_nopad(device, B, Lv, Le, Hq, Hkv, D):
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    L = Lv + Le
    inputs, cos, sin = make_inputs(B, Lv, Le, Hq, Hkv, D, device, seed=3)
    # joint mask: prefix (VLM) causal over blocks + suffix (expert) sees everything; here just causal for the test
    def causal(b, h, q, kv):
        return q >= kv

    bm337 = create_block_mask(causal, B, Hq, L, L, device=device, _compile=False)
    Lp = ((L + 127) // 128) * 128
    bm_pad = create_block_mask(causal, B, Hq, Lp, Lp, device=device, _compile=False)
    kernel_options = {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2}

    Q, K, V = joint_qkv_prep(*inputs, cos, sin)
    out = flex_attention(Q, K, V, block_mask=bm337, enable_gqa=True, scale=D ** -0.5, kernel_options=kernel_options)
    Qp, Kp, Vp = joint_qkv_prep_reference(*inputs, cos, sin, pad_to=Lp)
    out_p = flex_attention(Qp, Kp, Vp, block_mask=bm_pad, enable_gqa=True, scale=D ** -0.5, kernel_options=kernel_options)
    compare("flex no-pad vs pad: forward", [out], [out_p[:, :, :L]], 1e-4, 1e-4)
    up = torch.randn_like(out)
    g = torch.autograd.grad((out * up).sum(), inputs)
    gp = torch.autograd.grad((out_p[:, :, :L] * up).sum(), inputs)
    # The bf16 (VLM) grads are fp32 grads rounded once: an fp32-level difference upstream (here the eager flex backward
    # with vs without the pad) can move an element to the neighbouring bf16 value, so they get one bf16 rounding step
    # (rtol 2**-7); the fp32 (expert) grads keep 1e-4.
    is32 = [x.dtype == torch.float32 for x in inputs]
    compare("flex no-pad vs pad: input grads fp32", [a for a, f in zip(g, is32) if f],
            [b for b, f in zip(gp, is32) if f], 1e-4, 1e-4)
    compare("flex no-pad vs pad: input grads bf16 (1 rounding step)", [a for a, f in zip(g, is32) if not f],
            [b for b, f in zip(gp, is32) if not f], 2 ** -7, 1e-4)


def run_flex_nopad_compiled(device, B, Lv, Le, Hq, Hkv, D):
    """Compiled flex (Triton kernels, as in training) with and without the pad, through the repo's helpers:
    build_block_mask + flex_attention_with_block_mask (padded to 384) vs build_block_mask(pad_to_block=False) +
    flex_attention_prepared (L = 337). run_flex_nopad calls flex_attention eagerly, i.e. PyTorch's reference
    implementation, which does not run the IS_DIVISIBLE=False kernels that dropping the pad relies on."""
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                        "lingbotvla", "models", "vla", "lingbot_vla", "flex_attention.py")
    spec = importlib.util.spec_from_file_location("_lingbot_flex_attention", path)
    fa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fa)

    L = Lv + Le
    g = torch.Generator(device="cpu").manual_seed(7)
    # training-like joint mask (make_att_2d_masks with vlm_causal): causal prefix right-padded to a random length,
    # then the state token and one block of action tokens that see the whole valid prefix
    valid = torch.randint(Lv // 2, Lv + 1, (B,), generator=g)
    pad = torch.cat([torch.arange(Lv)[None, :] < valid[:, None], torch.ones(B, Le, dtype=torch.bool)], dim=1)
    ar = torch.cat([torch.ones(B, Lv + 2, dtype=torch.int32), torch.zeros(B, Le - 2, dtype=torch.int32)], dim=1)
    cum = torch.cumsum(ar, dim=1)
    mask3d = ((cum[:, None, :] <= cum[:, :, None]) & pad[:, None, :] & pad[:, :, None]).to(device)

    bm_pad = fa.build_block_mask(mask3d, Hq, L, L)
    bm_nopad = fa.build_block_mask(mask3d, Hq, L, L, pad_to_block=False)
    padded = torch.compile(fa.flex_attention_with_block_mask)
    prepared = torch.compile(fa.flex_attention_prepared)

    xs = [torch.randn(B, L, h, D, generator=g).to(device).requires_grad_(True) for h in (Hq, Hkv, Hkv)]
    up = torch.randn(B, L, Hq * D, generator=g).to(device)
    out_p = padded(*xs, bm_pad, L)
    out = prepared(*[x.transpose(1, 2).contiguous() for x in xs], bm_nopad)
    compare("compiled flex no-pad vs pad: forward", [out], [out_p], 1e-4, 1e-4)
    gp = torch.autograd.grad((out_p * up).sum(), xs)
    gn = torch.autograd.grad((out * up).sum(), xs)
    compare("compiled flex no-pad vs pad: dQ/dK/dV", gn, gp, 1e-4, 1e-4)
    check("compiled flex no-pad: finite", all(bool(torch.isfinite(t).all()) for t in (out, *gn)))


def run_timing(device, B, Lv, Le, Hq, Hkv, D, iters=20):
    inputs, cos, sin = make_inputs(B, Lv, Le, Hq, Hkv, D, device, seed=5)
    L = Lv + Le
    Lp = ((L + 127) // 128) * 128
    upstream = [torch.randn(B, h, L, D, device=device) for h in (Hq, Hkv, Hkv)]
    upstream_p = [torch.nn.functional.pad(u, (0, 0, 0, Lp - L)) for u in upstream]

    def eager():
        o = joint_qkv_prep_reference(*inputs, cos, sin, pad_to=Lp)
        torch.autograd.grad(sum((x * u).sum() for x, u in zip(o, upstream_p)), inputs)

    def fused():
        o = joint_qkv_prep(*inputs, cos, sin)
        torch.autograd.grad(sum((x * u).sum() for x, u in zip(o, upstream)), inputs)

    for name, fn in (("eager cat+RoPE+pad (fwd+bwd)", eager), ("fused joint_qkv_prep (fwd+bwd)", fused)):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        print(f"  {name:36s} {(time.perf_counter() - t0) / iters * 1e3:8.3f} ms per layer")
    # forward only, with CUDA events
    for name, fn in (("eager forward", lambda: joint_qkv_prep_reference(*inputs, cos, sin, pad_to=Lp)),
                     ("fused forward", lambda: joint_qkv_prep(*inputs, cos, sin))):
        with torch.no_grad():
            for _ in range(3):
                fn()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                fn()
            e.record()
            torch.cuda.synchronize()
        print(f"  {name:36s} {s.elapsed_time(e) / iters * 1e3:8.1f} us per layer")


if __name__ == "__main__":
    # training shape: B=32, Lv=286, Le=51, Hq=32, Hkv=8, D=128 (robotwin_dist_muon.yaml, micro 32)
    if "--interpret" in sys.argv:
        # run the Triton kernels through the Triton CPU interpreter (TRITON_INTERPRET=1 must be set before import)
        import lingbotvla.ops.joint_qkv_prep as _m
        assert os.environ.get("TRITON_INTERPRET") == "1" and _m.triton is not None, "set TRITON_INTERPRET=1 and install triton"
        _m._use_triton = lambda t: True
        run_equivalence("cpu", 2, 13, 5, 4, 2, 16, rtol=0.0, atol=0.0, tag="interpreter small")
        run_equivalence("cpu", 1, 37, 11, 2, 1, 128, rtol=0.0, atol=0.0, tag="interpreter D=128")
        run_bf16_out("cpu", 2, 13, 5, 4, 2, 16, rtol=0.0, atol=0.0, tag="interpreter small")
        run_bf16_out("cpu", 1, 37, 11, 2, 1, 128, rtol=0.0, atol=0.0, tag="interpreter D=128")
        run_nonfinite("cpu", "interpreter")
        run_rounding_payloads("cpu", "interpreter")
        print("FAILED:" if FAILED else "ALL PASSED (interpreter)", FAILED or "")
        sys.exit(1 if FAILED else 0)
    run_equivalence("cpu", 2, 13, 5, 4, 2, 16, rtol=0.0, atol=0.0, tag="CPU small")
    run_bf16_out("cpu", 2, 13, 5, 4, 2, 16, rtol=0.0, atol=0.0, tag="CPU small")
    run_nonfinite("cpu", "CPU")
    run_gradcheck()
    if torch.cuda.is_available():
        dev = "cuda"
        # exact: the kernels are compiled with enable_fp_fusion=False, so no FMA contraction on either side
        run_equivalence(dev, 2, 13, 5, 4, 2, 16, rtol=0.0, atol=0.0, tag="GPU small")
        run_nonfinite(dev, "GPU")
        run_rounding_payloads(dev, "GPU")
        run_equivalence(dev, 32, 286, 51, 32, 8, 128, rtol=0.0, atol=0.0, tag="GPU training shape")
        run_bf16_out(dev, 32, 286, 51, 32, 8, 128, rtol=0.0, atol=0.0, tag="GPU training shape")
        run_flex_nopad(dev, 4, 286, 51, 32, 8, 128)
        run_flex_nopad_compiled(dev, 32, 286, 51, 32, 8, 128)
        run_timing(dev, 32, 286, 51, 32, 8, 128)
    else:
        print("CUDA not available: GPU checks skipped")
    if FAILED:
        print("FAILED:", FAILED)
        sys.exit(1)
    print("ALL PASSED")
