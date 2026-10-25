"""Tests for lingbotvla/ops/joint_fa2_attention.py.

    python profile/bench/test_joint_fa2_attention.py                                  # CPU: plan/mask, fp64 pipeline vs dense reference (+grads), bf16 smoke; GPU part if CUDA + flash_attn
    TRITON_INTERPRET=1 python profile/bench/test_joint_fa2_attention.py --interpret   # Triton pack/unpack kernels on the CPU interpreter vs the torch fallbacks (bit-exact)

CPU part needs only torch; the FA2 call is replaced by an equivalent torch varlen attention there.
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
    # machine without transformers/flash_attn (CPU-only test box): expose lingbotvla.ops as a bare namespace package
    import types
    for name, sub in (("lingbotvla", "lingbotvla"), ("lingbotvla.ops", "lingbotvla/ops")):
        m = types.ModuleType(name)
        m.__path__ = [os.path.join(_REPO, sub)]
        sys.modules[name] = m

from lingbotvla.ops import joint_fa2_attention as M  # noqa: E402
from lingbotvla.ops.joint_fa2_attention import (  # noqa: E402
    build_fa2_plan, check_plan_against_mask, joint_attention_reference, joint_fa2_attention, plan_to_dense_mask,
)

FAILED = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILED.append(name)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from lingbotvla/models/vla/lingbot_vla/utils.py:64-94 (big_vision cumsum scheme)."""
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def scenario(B, n_img, n_lang, n_query, Na, device, seed=0, pad_counts=None):
    """Token layout like the model: [img, lang(with pad at its tail), task queries] = prefix, [state, actions] = suffix.
    Returns pad_masks [B, L], att_masks [B, L] (mask_ar), Lv, Le."""
    g = torch.Generator().manual_seed(seed)
    Lv = n_img + n_lang + n_query
    Le = 1 + Na
    L = Lv + Le
    pad = torch.ones(B, L, dtype=torch.bool)
    if pad_counts is None:
        pad_counts = torch.randint(0, n_lang + 1, (B,), generator=g).tolist()
        pad_counts[0] = 0                          # one sample without pad
    for b, p in enumerate(pad_counts):
        if p > 0:
            pad[b, n_img + n_lang - p:n_img + n_lang] = False
    att = torch.zeros(B, L, dtype=torch.int64)
    att[:, :Lv] = 1                                # vlm_causal: prefix mask_ar all ones
    att[:, Lv] = 1                                 # state opens a segment
    att[:, Lv + 1] = 1                             # first action opens a segment, the rest share it
    return pad.to(device), att.to(device), Lv, Le


def rand_inputs(B, Lv, Le, Hq, Hkv, D, device, dtype_v, dtype_e, seed=0):
    g = torch.Generator().manual_seed(seed)
    L = Lv + Le
    mk = lambda *s, dt: (torch.randn(*s, generator=g) * 0.5).to(dt).to(device)  # noqa: E731
    q_v, k_v, v_v = mk(B, Lv, Hq, D, dt=dtype_v), mk(B, Lv, Hkv, D, dt=dtype_v), mk(B, Lv, Hkv, D, dt=dtype_v)
    q_e, k_e, v_e = mk(B, Le, Hq, D, dt=dtype_e), mk(B, Le, Hkv, D, dt=dtype_e), mk(B, Le, Hkv, D, dt=dtype_e)
    pos = torch.arange(L, dtype=torch.float64)[None, :].expand(B, L) + torch.randint(0, 50, (B, 1), generator=g)
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, dtype=torch.float64) / D))        # [D/2]
    ang = pos[:, :, None] * inv[None, None, :]                                     # [B, L, D/2]
    cdt = torch.float64 if dtype_e == torch.float64 else torch.float32
    cos = torch.cat([ang.cos(), ang.cos()], -1).to(cdt).to(device)                 # both halves identical (Qwen3)
    sin = torch.cat([ang.sin(), ang.sin()], -1).to(cdt).to(device)
    return q_v, k_v, v_v, q_e, k_e, v_e, cos, sin


# ---------------------------------------------------------------------------------------------- 1. plan vs mask
def test_plan_mask(device):
    for seed, (B, n_img, n_lang, n_query, Na) in enumerate([(3, 8, 8, 4, 5), (4, 6, 10, 2, 3), (2, 198, 72, 16, 50)]):
        pad, att, Lv, Le = scenario(B, n_img, n_lang, n_query, Na, device, seed=seed)
        mask = make_att_2d_masks(pad, att)
        plan = build_fa2_plan(pad, Lv, Le)
        dense = plan_to_dense_mask(plan)
        check(f"plan mask == make_att_2d_masks (B={B}, L={Lv + Le}, seed={seed})", torch.equal(dense, mask),
              f"visible {int(mask.sum())}")
        ok = True
        try:
            check_plan_against_mask(plan, mask)
        except RuntimeError as e:
            ok = False
            print("   ", e)
        check("check_plan_against_mask passes on the model mask", ok)
        bad = mask.clone()
        bad[0, Lv + 1, 0] = ~bad[0, Lv + 1, 0]
        raised = False
        try:
            check_plan_against_mask(plan, bad)
        except RuntimeError:
            raised = True
        check("check_plan_against_mask raises on a 1-bit change", raised)
        # structural facts the design relies on
        cuA, cuBk = plan.cuA.tolist(), plan.cuBk.tolist()
        lenA = [cuA[i + 1] - cuA[i] for i in range(B)]
        npad = (~pad).sum(1).tolist()
        check("len_A[b] == Lv + 1 - pad_b", all(lenA[b] == Lv + 1 - npad[b] for b in range(B)))
        check("cuBk[b+1]-cuBk[b] == len_A[b] + Na", all(cuBk[b + 1] - cuBk[b] == lenA[b] + Na for b in range(B)))
        check("rowsA / rowsBk are upper bounds", cuA[-1] <= plan.rowsA and cuBk[-1] <= plan.rowsBk)


# ---------------------------------------------------------------------------------------------- 2. fp64 pipeline vs dense reference
def test_pipeline_fp64(device):
    B, n_img, n_lang, n_query, Na, Hq, Hkv, D = 3, 8, 8, 4, 5, 4, 2, 8
    pad, att, Lv, Le = scenario(B, n_img, n_lang, n_query, Na, device, seed=1)
    mask = make_att_2d_masks(pad, att)
    plan = build_fa2_plan(pad, Lv, Le)
    ins = rand_inputs(B, Lv, Le, Hq, Hkv, D, device, torch.float64, torch.float64, seed=1)
    ins = [x.clone().requires_grad_(True) for x in ins[:6]] + list(ins[6:])
    ref_v, ref_e = joint_attention_reference(*ins, mask, vlm_dtype=torch.float64, exp_dtype=torch.float64)
    out_v, out_e = joint_fa2_attention(*ins, plan, attn_dtype=torch.float64, vlm_dtype=torch.float64, exp_dtype=torch.float64)
    dv = (out_v - ref_v).abs().max().item()
    de = (out_e - ref_e).abs().max().item()
    check("fp64 forward att_vlm == dense reference", dv < 1e-12, f"max abs diff {dv:.2e}")
    check("fp64 forward att_exp == dense reference", de < 1e-12, f"max abs diff {de:.2e}")
    pad_rows = out_v[~pad[:, :Lv]]
    check("pad query rows of att_vlm are exactly 0", pad_rows.numel() > 0 and bool((pad_rows == 0).all()))
    g = torch.Generator().manual_seed(7)
    gv = torch.randn(out_v.shape, generator=g, dtype=torch.float64).to(device)
    ge = torch.randn(out_e.shape, generator=g, dtype=torch.float64).to(device)
    grads_out = torch.autograd.grad((out_v, out_e), ins[:6], (gv, ge))
    grads_ref = torch.autograd.grad((ref_v, ref_e), ins[:6], (gv, ge))
    names = ["q_vlm", "k_vlm", "v_vlm", "q_exp", "k_exp", "v_exp"]
    for n, a, r in zip(names, grads_out, grads_ref):
        dd = (a - r).abs().max().item()
        check(f"fp64 grad {n} == dense reference", dd < 1e-11, f"max abs diff {dd:.2e}, |ref| max {r.abs().max().item():.2e}")
    # pad tokens must receive zero gradient (they are neither keys nor useful queries)
    gk = grads_out[1][~pad[:, :Lv]]
    check("pad tokens get zero k gradient", bool((gk == 0).all()))
    # gradcheck of the whole pipeline on a tiny problem
    pad2, att2, Lv2, Le2 = scenario(2, 3, 3, 1, 2, device, seed=3, pad_counts=[0, 2])
    plan2 = build_fa2_plan(pad2, Lv2, Le2)
    ins2 = rand_inputs(2, Lv2, Le2, 2, 1, 4, device, torch.float64, torch.float64, seed=3)
    ins2 = [x.clone().requires_grad_(True) for x in ins2[:6]] + list(ins2[6:])
    fn = lambda *xs: joint_fa2_attention(*xs, ins2[6], ins2[7], plan2, attn_dtype=torch.float64, vlm_dtype=torch.float64, exp_dtype=torch.float64)  # noqa: E731
    ok = torch.autograd.gradcheck(fn, tuple(ins2[:6]), eps=1e-6, atol=1e-6, rtol=1e-5)
    check("gradcheck (fp64, tiny)", ok)


# ---------------------------------------------------------------------------------------------- 3. bf16 smoke (CPU or GPU)
def test_bf16(device, tag):
    B, n_img, n_lang, n_query, Na, Hq, Hkv, D = 2, 8, 8, 4, 5, 4, 2, 16
    pad, att, Lv, Le = scenario(B, n_img, n_lang, n_query, Na, device, seed=2)
    mask = make_att_2d_masks(pad, att)
    plan = build_fa2_plan(pad, Lv, Le)
    ins = rand_inputs(B, Lv, Le, Hq, Hkv, D, device, torch.bfloat16, torch.float32, seed=2)
    out_v, out_e = joint_fa2_attention(*ins, plan)
    ref_v, ref_e = joint_attention_reference(*ins, mask, vlm_dtype=torch.float32, exp_dtype=torch.float32)
    check(f"[{tag}] bf16 dtypes/shapes", out_v.dtype == torch.bfloat16 and out_e.dtype == torch.float32
          and out_v.shape == (B, Lv, Hq * D) and out_e.shape == (B, Le, Hq * D))
    rv = ((out_v.float() - ref_v).abs().max() / ref_v.abs().max()).item()
    re = ((out_e - ref_e).abs().max() / ref_e.abs().max()).item()
    check(f"[{tag}] bf16 att_vlm within bf16 tolerance of fp32 reference", rv < 3e-2, f"rel max err {rv:.2e}")
    check(f"[{tag}] bf16 att_exp within bf16 tolerance of fp32 reference", re < 3e-2, f"rel max err {re:.2e}")
    check(f"[{tag}] bf16 pad rows are 0", bool((out_v[~pad[:, :Lv]] == 0).all()))


# ---------------------------------------------------------------------------------------------- 4. GPU: FA2 vs reference, vs flex-bf16, timing
def test_gpu():
    dev = "cuda"
    if M._fa2_varlen_func is None:
        print("SKIP GPU FA2 tests: flash_attn not importable")
        return
    # training shape: 3 images x 66 + lang 72 + 2 x 8 queries = 286 prefix; state + 50 actions
    B, n_img, n_lang, n_query, Na, Hq, Hkv, D = 32, 198, 72, 16, 50, 32, 8, 128
    pad, att, Lv, Le = scenario(B, n_img, n_lang, n_query, Na, dev, seed=9)
    mask = make_att_2d_masks(pad, att)
    plan = build_fa2_plan(pad, Lv, Le)
    check_plan_against_mask(plan, mask)
    ins = rand_inputs(B, Lv, Le, Hq, Hkv, D, dev, torch.bfloat16, torch.float32, seed=9)
    ins = [x.clone().requires_grad_(True) for x in ins[:6]] + list(ins[6:])
    ref_v, ref_e = joint_attention_reference(*ins, mask, vlm_dtype=torch.float32, exp_dtype=torch.float32)
    out_v, out_e = joint_fa2_attention(*ins, plan)
    # the same inputs through flex in bf16 (what --train.attention_dtype=bf16 does), for an apples-to-apples error
    from lingbotvla.ops.joint_qkv_prep import joint_qkv_prep
    from lingbotvla.models.vla.lingbot_vla.flex_attention import build_block_mask, flex_attention_prepared
    Q, K, V = joint_qkv_prep(*ins[:6], ins[6], ins[7], out_dtype=torch.bfloat16)
    bm = build_block_mask(mask, Hq, Lv + Le, Lv + Le, pad_to_block=False)
    flex_o = flex_attention_prepared(Q, K, V, bm)                              # [B, L, Hq*D] bf16
    flex_v, flex_e = flex_o[:, :Lv].float(), flex_o[:, Lv:].float()
    valid_v = pad[:, :Lv]
    e_fa2_v = ((out_v.float() - ref_v).abs()[valid_v].max() / ref_v.abs().max()).item()
    e_flx_v = ((flex_v - ref_v).abs()[valid_v].max() / ref_v.abs().max()).item()
    e_fa2_e = ((out_e - ref_e).abs().max() / ref_e.abs().max()).item()
    e_flx_e = ((flex_e - ref_e).abs().max() / ref_e.abs().max()).item()
    print(f"    rel max err vs fp32 reference: att_vlm FA2 {e_fa2_v:.3e} | flex-bf16 {e_flx_v:.3e};  att_exp FA2 {e_fa2_e:.3e} | flex-bf16 {e_flx_e:.3e}")
    check("[GPU] FA2 att_vlm error <= 1.5x flex-bf16 error", e_fa2_v <= 1.5 * e_flx_v + 1e-4)
    check("[GPU] FA2 att_exp error <= 1.5x flex-bf16 error", e_fa2_e <= 1.5 * e_flx_e + 1e-4)
    check("[GPU] FA2 pad rows are 0", bool((out_v[~valid_v] == 0).all()))
    g = torch.Generator().manual_seed(13)
    gv = torch.randn(out_v.shape, generator=g).to(torch.bfloat16).to(dev)
    ge = torch.randn(out_e.shape, generator=g).to(dev)
    grads = torch.autograd.grad((out_v, out_e), ins[:6], (gv, ge))
    grads_ref = torch.autograd.grad((ref_v, ref_e), ins[:6], (gv.float(), ge))
    grads_flx = torch.autograd.grad(flex_o, ins[:6], torch.cat([gv, ge.to(torch.bfloat16)], 1))
    for n, a, f, r in zip(["q_vlm", "k_vlm", "v_vlm", "q_exp", "k_exp", "v_exp"], grads, grads_flx, grads_ref):
        ea = ((a.float() - r.float()).abs().max() / r.float().abs().max()).item()
        ef = ((f.float() - r.float()).abs().max() / r.float().abs().max()).item()
        check(f"[GPU] grad {n}: FA2 err {ea:.3e} <= 1.5x flex-bf16 err {ef:.3e}", ea <= 1.5 * ef + 1e-4)

    def bench(fn, iters=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    xs = [x.detach().requires_grad_(True) for x in ins[:6]]
    t_fa2_f = bench(lambda: joint_fa2_attention(*xs, ins[6], ins[7], plan))
    t_flx_f = bench(lambda: flex_attention_prepared(*joint_qkv_prep(*xs, ins[6], ins[7], out_dtype=torch.bfloat16), bm))

    def fa2_fb():
        ov, oe = joint_fa2_attention(*xs, ins[6], ins[7], plan)
        torch.autograd.backward((ov, oe), (gv, ge))

    def flx_fb():
        o = flex_attention_prepared(*joint_qkv_prep(*xs, ins[6], ins[7], out_dtype=torch.bfloat16), bm)
        o.backward(torch.cat([gv, ge.to(torch.bfloat16)], 1))

    t_fa2_fb = bench(fa2_fb)
    t_flx_fb = bench(flx_fb)
    print(f"    training shape, per layer: FA2 path fwd {t_fa2_f:.3f} ms, fwd+bwd {t_fa2_fb:.3f} ms  |  flex-bf16 (prep+flex) fwd {t_flx_f:.3f} ms, fwd+bwd {t_flx_fb:.3f} ms")


# ---------------------------------------------------------------------------------------------- 5. kernels vs torch fallbacks (interpreter or GPU)
def test_kernels_vs_torch(device, B, n_img, n_lang, n_query, Na, Hq, Hkv, D, tag, exact=True):
    pad, att, Lv, Le = scenario(B, n_img, n_lang, n_query, Na, device, seed=5)
    plan = build_fa2_plan(pad, Lv, Le)
    L = Lv + Le
    ins = rand_inputs(B, Lv, Le, Hq, Hkv, D, device, torch.bfloat16, torch.float32, seed=5)
    q_v, k_v, v_v, q_e, k_e, v_e, cos, sin = ins
    nA, nBk = plan.cuA[-1].item(), plan.cuBk[-1].item()

    def cmp(name, a, b):
        # exact: bit-identical. The kernels are compiled with enable_fp_fusion=False, so they equal the torch
        # fallbacks on the GPU as well (and in the interpreter).
        if exact:
            ok = a.dtype == b.dtype and torch.equal(a, b)
            det = "" if ok else f"max abs diff {(a.float() - b.float()).abs().max().item():.3e}"
        else:
            af, bf = a.float(), b.float()
            d = (af - bf).abs().max().item()
            scale = max(1.0, bf.abs().max().item())
            if b.dtype == torch.bfloat16:
                frac = (a != b).float().mean().item()
                ok = a.dtype == b.dtype and frac <= 1e-3 and d <= 2 ** -7 * scale
                det = f"mismatch frac {frac:.2e}, max abs diff {d:.3e}"
            else:
                ok = a.dtype == b.dtype and d <= 1e-5 * scale
                det = f"max abs diff {d:.3e}"
        check(f"[{tag}] {name}", ok, det)

    for out_dtype in (torch.bfloat16, torch.float32):
        # pack forward, q pattern (A + Bq) and k pattern (A + Bk)
        for nm, xv, xe, d2, rows2, n2, rope in (("q", q_v, q_e, plan.dstBq, plan.rowsBq, plan.rowsBq, True),
                                                 ("k", k_v, k_e, plan.dstBk, plan.rowsBk, nBk, True),
                                                 ("v", v_v, v_e, plan.dstBk, plan.rowsBk, nBk, False)):
            t1, t2 = M._pack_fwd_triton(xv, xe, cos, sin, plan.dstA, d2, plan.rowsA, rows2, rope, out_dtype)
            r1, r2 = M._pack_fwd_torch(xv, xe, cos, sin, plan.dstA, d2, plan.rowsA, rows2, rope, out_dtype)
            cmp(f"pack fwd {nm} A rows ({out_dtype})", t1[:nA], r1[:nA])
            cmp(f"pack fwd {nm} B rows ({out_dtype})", t2[:n2], r2[:n2])
        # pack backward
        g = torch.Generator().manual_seed(11)
        for nm, H_, d2, rows2, rope in (("q", Hq, plan.dstBq, plan.rowsBq, True), ("k", Hkv, plan.dstBk, plan.rowsBk, True),
                                        ("v", Hkv, plan.dstBk, plan.rowsBk, False)):
            g1 = torch.randn(plan.rowsA, H_, D, generator=g).to(out_dtype).to(device)
            g2 = torch.randn(rows2, H_, D, generator=g).to(out_dtype).to(device)
            tv, te = M._pack_bwd_triton(g1, g2, cos, sin, plan.dstA, d2, Lv, L, H_, D, rope, torch.bfloat16, torch.float32)
            rv, re = M._pack_bwd_torch(g1, g2, cos, sin, plan.dstA, d2, Lv, L, H_, D, rope, torch.bfloat16, torch.float32)
            cmp(f"pack bwd {nm} dv ({out_dtype} grads)", tv, rv)
            cmp(f"pack bwd {nm} de ({out_dtype} grads)", te, re)
        # unpack forward / backward
        o1 = torch.randn(plan.rowsA, Hq, D, generator=g).to(out_dtype).to(device)
        o2 = torch.randn(plan.rowsBq, Hq, D, generator=g).to(out_dtype).to(device)
        for exp_dtype in (torch.float32, torch.bfloat16):
            tv, te = M._unpack_fwd_triton(o1, o2, plan.dstA, plan.dstBq, Lv, Le, torch.bfloat16, exp_dtype)
            rv, re = M._unpack_fwd_torch(o1, o2, plan.dstA, plan.dstBq, Lv, Le, torch.bfloat16, exp_dtype)
            cmp(f"unpack fwd att_vlm ({out_dtype} in, exp {exp_dtype})", tv, rv)
            cmp(f"unpack fwd att_exp ({out_dtype} in, exp {exp_dtype})", te, re)
        gv = torch.randn(B, Lv, Hq * D, generator=g).to(torch.bfloat16).to(device)
        ge = torch.randn(B, Le, Hq * D, generator=g).to(device)
        t1, t2 = M._unpack_bwd_triton(gv, ge, plan.dstA, plan.dstBq, plan.rowsA, plan.rowsBq, Hq, D, out_dtype)
        r1, r2 = M._unpack_bwd_torch(gv, ge, plan.dstA, plan.dstBq, plan.rowsA, plan.rowsBq, Hq, D, out_dtype)
        cmp(f"unpack bwd dO_A ({out_dtype})", t1[:nA], r1[:nA])
        cmp(f"unpack bwd dO_B ({out_dtype})", t2, r2)


if __name__ == "__main__":
    if "--interpret" in sys.argv:
        assert os.environ.get("TRITON_INTERPRET") == "1", "set TRITON_INTERPRET=1 before running with --interpret"
        M._use_triton = lambda t: True
        test_kernels_vs_torch("cpu", 3, 8, 8, 4, 5, 4, 2, 8, tag="interpreter small", exact=True)
        test_kernels_vs_torch("cpu", 2, 20, 9, 4, 6, 2, 1, 128, tag="interpreter D=128", exact=True)
        print("FAILED:" if FAILED else "ALL PASSED (interpreter)", FAILED or "")
        sys.exit(1 if FAILED else 0)

    test_plan_mask("cpu")
    test_pipeline_fp64("cpu")
    test_bf16("cpu", "CPU")
    if torch.cuda.is_available():
        test_plan_mask("cuda")
        test_kernels_vs_torch("cuda", 3, 8, 8, 4, 5, 4, 2, 8, tag="GPU kernels small", exact=True)
        test_kernels_vs_torch("cuda", 4, 198, 72, 16, 50, 32, 8, 128, tag="GPU kernels D=128", exact=True)
        test_bf16("cuda", "GPU torch-attn" if M._FORCE_TORCH_ATTN else "GPU")
        test_gpu()
    print("FAILED:" if FAILED else "ALL PASSED", FAILED or "")
    sys.exit(1 if FAILED else 0)
