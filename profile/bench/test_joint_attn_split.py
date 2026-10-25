"""Correctness and timing tests for lingbotvla/ops/joint_attn_split.py.

    python profile/bench/test_joint_attn_split.py
    TRITON_INTERPRET=1 python profile/bench/test_joint_attn_split.py --interpret

Compares joint_attn_split with the eager chain (joint_attn_split_reference) forward and backward; everything is a
pure copy plus one RTNE rounding, so bit-exact equality is required on CPU, interpreter and GPU.  Covers L_pad > L
(the current padded flex output) and L_pad == L (after the pad is removed).  Times eager vs fused at the training
shape (B=32, Hq=32, L=337 / L_pad=384, D=128) on GPU.  NaN/Inf in the flex output and in the upstream gradients must
be carried through like every other value.
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
from lingbotvla.ops.joint_attn_split import joint_attn_split, joint_attn_split_reference  # noqa: E402

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
    from lingbotvla.ops.joint_attn_split import _rtne_bf16_bits as _rtne_helper  # noqa: E402
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


def run_case(device, B, H, Lv, Le, Lp, D, tag, o_dtype=torch.float32, exp_dtype=None):
    """o_dtype=bf16 + exp_dtype=fp32 is the --train.attention_dtype=bf16 configuration (expert output upcast)."""
    gen = torch.Generator(device="cpu").manual_seed(0)
    o = torch.randn(B, H, Lp, D, generator=gen).to(device=device, dtype=o_dtype).requires_grad_(True)
    gv = torch.randn(B, Lv, H * D, generator=gen).to(device=device, dtype=torch.bfloat16)
    ge = torch.randn(B, Le, H * D, generator=gen).to(device=device, dtype=exp_dtype or o_dtype)
    rv, re_ = joint_attn_split_reference(o, Lv, Le, exp_dtype=exp_dtype)
    (rgo,) = torch.autograd.grad((rv.float() * gv.float()).sum() + (re_.float() * ge.float()).sum(), (o,))
    v, e = joint_attn_split(o, Lv, Le, exp_dtype=exp_dtype)
    (go,) = torch.autograd.grad((v.float() * gv.float()).sum() + (e.float() * ge.float()).sum(), (o,))
    exp_dt = exp_dtype or o_dtype
    check(f"{tag} forward vlm (bf16) bit-exact", v.dtype == torch.bfloat16 and torch.equal(v, rv), f"shape {tuple(v.shape)}")
    check(f"{tag} forward expert ({str(exp_dt).replace('torch.', '')}) bit-exact", e.dtype == exp_dt and torch.equal(e, re_), f"shape {tuple(e.shape)}")
    check(f"{tag} backward dO bit-exact (incl. zero pad rows)", go.shape == rgo.shape and go.dtype == o_dtype and torch.equal(go, rgo),
          f"max|diff|={(go.float() - rgo.float()).abs().max().item():.3e}")
    check(f"{tag} outputs contiguous", v.is_contiguous() and e.is_contiguous() and go.is_contiguous())


def run_nonfinite(device, tag):
    """NaN/Inf in O (forward) and in the two upstream gradients (backward) come out where the eager chain has them."""
    B, H, Lv, Le, Lp, D = 2, 4, 13, 5, 18, 16
    gen = torch.Generator(device="cpu").manual_seed(2)
    o = torch.randn(B, H, Lp, D, generator=gen)
    o[0, 0, 3, 1], o[1, 1, 2, 9], o[0, 2, Lv + 1, 4] = float("nan"), float("inf"), -float("inf")
    # NaN payloads in VLM rows whose +0x7FFF carry overflows in the integer rounding (float('nan') does not)
    ob = o.view(torch.int32)
    ob[0, 0, 4, 2], ob[1, 0, 5, 3], ob[1, 2, 6, 5] = 0x7FFFFFFF, 0x7F800001, -1  # -1 = 0xFFFFFFFF
    gv = torch.randn(B, Lv, H * D, generator=gen)
    ge = torch.randn(B, Le, H * D, generator=gen)
    gv[0, 5, 7], gv[1, 2, 3], ge[0, 1, 2] = float("nan"), float("inf"), float("nan")
    o = o.to(device).requires_grad_(True)
    gv, ge = gv.to(device=device, dtype=torch.bfloat16), ge.to(device)
    rv, re_ = joint_attn_split_reference(o, Lv, Le)
    (rgo,) = torch.autograd.grad((rv.float() * gv.float()).sum() + (re_ * ge).sum(), (o,))
    v, e = joint_attn_split(o, Lv, Le)
    (go,) = torch.autograd.grad((v.float() * gv.float()).sum() + (e * ge).sum(), (o,))
    ok, n = True, 0
    for a, b in ((v, rv), (e, re_), (go, rgo)):
        a, b = a.detach(), b.detach()
        nan = torch.isnan(b)
        ok = ok and torch.equal(torch.isnan(a), nan) and torch.equal(a[~nan], b[~nan])
        n += int(nan.sum()) + int(torch.isinf(b).sum())
    check(f"{tag} non-finite values carried through", ok, f"({n} NaN/Inf elements, all where eager has them)")


def run_timing(device, B, H, Lv, Le, Lp, D, iters=20):
    o = torch.randn(B, H, Lp, D, device=device, requires_grad=True)
    gv = torch.randn(B, Lv, H * D, device=device, dtype=torch.bfloat16)
    ge = torch.randn(B, Le, H * D, device=device)
    for name, fn in (("eager slice/transpose/cast/slice", joint_attn_split_reference), ("fused joint_attn_split", joint_attn_split)):
        def step():
            v, e = fn(o, Lv, Le)
            torch.autograd.grad((v.float() * gv.float()).sum() + (e * ge).sum(), (o,))
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        with torch.no_grad():
            s, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                fn(o, Lv, Le)
            en.record(); torch.cuda.synchronize()
        fwd = s.elapsed_time(en) / iters * 1e3
        t0 = time.perf_counter()
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
        print(f"  {name:36s} forward {fwd:8.1f} us   forward+backward {(time.perf_counter() - t0) / iters * 1e6:8.1f} us   (per layer, L_pad={Lp})")


if __name__ == "__main__":
    if "--interpret" in sys.argv:
        import lingbotvla.ops.joint_attn_split as _m
        assert os.environ.get("TRITON_INTERPRET") == "1" and _m.triton is not None, "set TRITON_INTERPRET=1 and install triton"
        _m._use_triton = lambda t: True
        run_case("cpu", 2, 4, 13, 5, 18, 16, "interpreter no-pad")
        run_case("cpu", 2, 4, 13, 5, 32, 16, "interpreter padded")
        run_case("cpu", 1, 2, 37, 11, 48, 128, "interpreter D=128")
        run_case("cpu", 2, 4, 13, 5, 18, 16, "interpreter bf16 O / fp32 expert", o_dtype=torch.bfloat16, exp_dtype=torch.float32)
        run_case("cpu", 1, 2, 37, 11, 48, 128, "interpreter bf16 O / fp32 expert D=128", o_dtype=torch.bfloat16, exp_dtype=torch.float32)
        run_nonfinite("cpu", "interpreter")
        run_rounding_payloads("cpu", "interpreter")
        print("FAILED:" if FAILED else "ALL PASSED (interpreter)", FAILED or "")
        sys.exit(1 if FAILED else 0)
    run_case("cpu", 2, 4, 13, 5, 18, 16, "CPU no-pad")
    run_case("cpu", 2, 4, 13, 5, 32, 16, "CPU padded")
    run_case("cpu", 2, 4, 13, 5, 18, 16, "CPU bf16 O / fp32 expert", o_dtype=torch.bfloat16, exp_dtype=torch.float32)
    run_nonfinite("cpu", "CPU")
    if torch.cuda.is_available():
        run_nonfinite("cuda", "GPU")
        run_rounding_payloads("cuda", "GPU")
        run_case("cuda", 32, 32, 286, 51, 384, 128, "GPU training shape, padded")
        run_case("cuda", 32, 32, 286, 51, 337, 128, "GPU training shape, no pad")
        run_case("cuda", 32, 32, 286, 51, 337, 128, "GPU training shape, bf16 O / fp32 expert", o_dtype=torch.bfloat16, exp_dtype=torch.float32)
        run_timing("cuda", 32, 32, 286, 51, 384, 128)
        run_timing("cuda", 32, 32, 286, 51, 337, 128)
    else:
        print("CUDA not available: GPU checks skipped")
    if FAILED:
        print("FAILED:", FAILED)
        sys.exit(1)
    print("ALL PASSED")
