"""Unit tests and micro-benchmarks for --train.moe_glue_opt (lingbotvla/ops/moe_glue.py and the fp32-B grouped GEMM).
One GPU:

    CUDA_VISIBLE_DEVICES=0 python profile/bench/test_moe_glue.py [--no-bench]

Tests:
  (a) K2b route_index: scatter_index == argsort(stable=True).argsort(), cumsum_M == cumsum(expert_histogram), the
      scattered gate weights == index_put, bitwise, on random routings (some experts without tokens, skewed loads,
      sizes up to the one-CTA limit); larger inputs must return None.
  (b) K2a group_gemm_same_nk with fp32 B == with B cast to bf16 beforehand, bitwise (forward and dgrad layouts).
  (c) K2a fp32 wgrad vs the bf16 wgrad .float(): relative difference <= 1e-2, and not farther from an fp64 reference.
  (d) fused_moe_forward with every glue option vs without: forward and the hidden / gate gradients bitwise, the weight
      gradients within (c)'s tolerance.
  (e) K2c router_tail: routing weights and the gradient of the router logits (through sigmoid) bitwise equal to the
      eager chain and its autograd, ties and signed zeros included; expert counts exact; the eager row-sum order; the
      kernels' IEEE add / mul / div keep denormals.
  Then timings of each piece (CUDA events, median of 50, inputs rotated so the weights do not stay in L2).
"""
import os
import statistics
import sys
import types

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lingbotvla.ops import moe_glue  # noqa: E402
from lingbotvla.ops.fused_moe import fused_moe_forward  # noqa: E402
from lingbotvla.ops.group_gemm.kernel.group_gemm import group_gemm_same_mn, group_gemm_same_nk  # noqa: E402
from lingbotvla.ops.group_gemm.kernel.moe import expert_histogram  # noqa: E402

FAILED = []
DEV = "cuda"
E, TOPK, HID, FFN = 32, 4, 768, 512       # the action expert's MoE: 32 experts, top-4, hidden 768, expert ffn 512
T = 1632                                  # tokens per rank and layer (32 samples x 51), 6528 routed rows


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILED.append(name)


def bits_equal(a, b):
    """Same shape, dtype and bit pattern (NaN payloads and signed zeros included)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.contiguous().view(-1).view(torch.uint8), b.contiguous().view(-1).view(torch.uint8))


def ndiff(a, b):
    return int((a.contiguous().view(-1).view(torch.uint8) != b.contiguous().view(-1).view(torch.uint8)).sum())


def random_routing(t, skew, gen, zero_experts=()):
    """[t, TOPK] distinct experts per row, skewed loads, some experts never chosen."""
    logits = torch.randn(t, E, generator=gen, device=DEV) + skew * torch.linspace(-2, 2, E, device=DEV)
    for e in zero_experts:
        logits[:, e] = -1e9
    return torch.topk(logits, TOPK, dim=-1).indices


# ------------------------------------------------------------------------------------------------ (a) K2b
def test_route_index():
    gen = torch.Generator(device=DEV).manual_seed(0)
    worst = 0
    cases = [(T, 0.0, ()), (T, 3.0, (0, 5, 31)), (1, 0.0, ()), (7, 1.0, ()), (100, 5.0, (1, 2, 3)),
             (1024, 0.0, tuple(range(16))), (2048, 2.0, ()), (4096, 0.0, (31,)), (16384 // TOPK, 1.0, (7,))]
    for _ in range(40):
        t = int(torch.randint(1, 16384 // TOPK + 1, (1,), generator=gen, device=DEV))
        cases.append((t, float(torch.rand(1, generator=gen, device=DEV)) * 6,
                      tuple(int(x) for x in torch.randint(0, E, (int(torch.randint(0, 8, (1,), generator=gen, device=DEV)),),
                                                          generator=gen, device=DEV))))
    for t, skew, zeros in cases:
        idx = random_routing(t, skew, gen, zeros)
        gate = torch.rand(t, TOPK, generator=gen, device=DEV).to(torch.bfloat16)
        pos, cumsum, sgate = moe_glue.route_index(idx, gate, E)
        ref_pos = idx.flatten().argsort(stable=True).argsort().int().view(idx.shape)
        ref_cumsum = torch.cumsum(expert_histogram(idx, E), dim=0)
        ref_sgate = torch.empty_like(gate.reshape(-1, 1))
        ref_sgate[ref_pos.flatten()] = gate.reshape(-1, 1)
        ok = bits_equal(pos, ref_pos) and bits_equal(cumsum, ref_cumsum) and bits_equal(sgate, ref_sgate)
        worst += not ok
        if not ok:
            print(f"   mismatch t={t} skew={skew} zeros={zeros}: pos {ndiff(pos, ref_pos)}, cumsum "
                  f"{bits_equal(cumsum, ref_cumsum)}, sgate {ndiff(sgate, ref_sgate)}")
    check("(a) K2b route_index == argsort(stable).argsort() / cumsum(histogram) / index_put, bitwise",
          worst == 0, f"({len(cases)} routings, sizes 1..{16384 // TOPK} tokens, {worst} failing)")
    idx = random_routing(16384 // TOPK + 1, 0.0, gen)
    check("(a) K2b route_index returns None above the one-CTA limit",
          moe_glue.route_index(idx, torch.zeros(idx.shape, device=DEV, dtype=torch.bfloat16), E) is None)


# ------------------------------------------------------------------------------------------------ (b), (c) K2a
def make_cumsum(gen, t=T):
    idx = random_routing(t, 2.0, gen, (3,))
    return torch.cumsum(expert_histogram(idx, E), dim=0), t * TOPK


def test_fp32_b_gemm():
    gen = torch.Generator(device=DEV).manual_seed(1)
    cumsum, m = make_cumsum(gen)
    ok_all = True
    for name, a_k, b_shape, transpose_b in [("forward fc1  (a [M,768], b [E,512,768]^T)", HID, (E, FFN, HID), True),
                                            ("forward fc2  (a [M,512], b [E,768,512]^T)", FFN, (E, HID, FFN), True),
                                            ("dgrad fc2    (a [M,768], b [E,768,512])", HID, (E, HID, FFN), False),
                                            ("dgrad fc1    (a [M,512], b [E,512,768])", FFN, (E, FFN, HID), False)]:
        a = (torch.randn(m, a_k, generator=gen, device=DEV) * 0.5).to(torch.bfloat16)
        b32 = torch.randn(*b_shape, generator=gen, device=DEV) * 0.02
        out32 = group_gemm_same_nk(a=a, b=b32, cumsum_M=cumsum, max_M=m, transpose_a=False, transpose_b=transpose_b)
        out16 = group_gemm_same_nk(a=a, b=b32.to(torch.bfloat16), cumsum_M=cumsum, max_M=m, transpose_a=False,
                                   transpose_b=transpose_b)
        ok = bits_equal(out32, out16)
        ok_all &= ok
        print(f"   {name}: {'bitwise equal' if ok else f'{ndiff(out32, out16)} elements differ'}")
    check("(b) K2a group_gemm_same_nk, fp32 B == B cast to bf16 first, bitwise", ok_all)


def test_fp32_wgrad():
    gen = torch.Generator(device=DEV).manual_seed(2)
    cumsum, m = make_cumsum(gen)
    for name, ma, nb in [("fc2 wgrad [E,768,512]", HID, FFN), ("fc1 wgrad [E,512,768]", FFN, HID)]:
        a = (torch.randn(m, ma, generator=gen, device=DEV) * 1e-2).to(torch.bfloat16)   # grad_out, transposed in use
        b = (torch.randn(m, nb, generator=gen, device=DEV)).to(torch.bfloat16)          # activations
        c32 = torch.empty(E, ma, nb, device=DEV, dtype=torch.float32)
        c16 = torch.empty(E, ma, nb, device=DEV, dtype=torch.bfloat16)
        for c in (c32, c16):
            group_gemm_same_mn(a=a, b=b, c=c, cumsum_K=cumsum, max_K=m, transpose_a=True, transpose_b=False)
        starts = torch.cat([cumsum.new_zeros(1), cumsum[:-1]]).tolist()
        ref = torch.stack([a[s:e].double().T @ b[s:e].double() for s, e in zip(starts, cumsum.tolist())])
        scale = ref.abs().max()
        rel = ((c32.double() - c16.double()).abs().max() / scale).item()
        err32 = ((c32.double() - ref).abs().max() / scale).item()
        err16 = ((c16.double() - ref).abs().max() / scale).item()
        rms32 = ((c32.double() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        rms16 = ((c16.double() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        check(f"(c) K2a {name}: fp32 vs bf16->fp32 within 1e-2, closer to fp64", rel <= 1e-2 and err32 <= err16,
              f"(max |fp32 - bf16| / max|ref| {rel:.2e}; error vs fp64: max {err32:.2e} vs {err16:.2e}, "
              f"rms {rms32:.2e} vs {rms16:.2e})")


# ------------------------------------------------------------------------------------------------ (e) K2c
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _ieee_ops_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    m = i < n
    a = tl.load(a_ptr + i, mask=m, other=1.0)
    b = tl.load(b_ptr + i, mask=m, other=1.0)
    tl.store(out_ptr + i, moe_glue._add(a, b), mask=m)
    tl.store(out_ptr + n + i, moe_glue._mul(a, b), mask=m)
    tl.store(out_ptr + 2 * n + i, moe_glue._div(a, b), mask=m)


def eager_tail(block, scores, sel, out_dtype):
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import _update_moe_runtime_stats

    rw = scores.gather(1, sel)
    if block.training:
        _update_moe_runtime_stats(block, rw, sel)
    if block.norm_topk_prob:
        rw = rw / (rw.sum(dim=-1, keepdim=True) + 1e-20)
    if block.routed_scaling_factor != 1.0:
        rw = rw * block.routed_scaling_factor
    return rw.to(out_dtype)


def fake_block(norm=True, scale=4.0, training=True):
    return types.SimpleNamespace(
        training=training, norm_topk_prob=norm, routed_scaling_factor=scale, num_experts=E,
        tokens_per_expert=torch.zeros(E, device=DEV), avg_topk_sigmoid_score=torch.zeros(1, device=DEV),
        e_score_correction_bias=torch.zeros(E, device=DEV))


def run_router(logits, upstream, use_glue, norm, scale, out_dtype, activation, bias):
    block = fake_block(norm, scale)
    x = logits.detach().clone().requires_grad_(True)
    scores = x.sigmoid() if activation == "sigmoid" else F.softmax(x, dim=1, dtype=torch.float)
    choice = scores + bias.unsqueeze(0)
    _, sel = torch.topk(choice, TOPK, dim=-1)
    rw = moe_glue.router_tail(block, scores, sel, out_dtype) if use_glue else None
    if rw is None:
        assert not use_glue, "router_tail declined a supported shape"
        rw = eager_tail(block, scores, sel, out_dtype)
    rw.backward(upstream.to(rw.dtype))
    return rw.detach(), sel, x.grad, block


def test_router_tail():
    gen = torch.Generator(device=DEV).manual_seed(3)
    # the eager row sum of [N, 4] fp32 is the tree (w0 + w1) + (w2 + w3)
    w = torch.rand(1 << 20, 4, generator=gen, device=DEV)
    s = w.sum(dim=-1)
    tree = (w[:, 0] + w[:, 1]) + (w[:, 2] + w[:, 3])
    seq = ((w[:, 0] + w[:, 1]) + w[:, 2]) + w[:, 3]
    other = (w[:, 0] + w[:, 2]) + (w[:, 1] + w[:, 3])
    check("(e) eager [N, 4].sum(-1) is (w0 + w1) + (w2 + w3)", bits_equal(s, tree),
          f"(differs from it in {ndiff(s, tree)} of {s.numel()} rows; from ((w0 + w1) + w2) + w3 in {ndiff(s, seq)}, "
          f"from (w0 + w2) + (w1 + w3) in {ndiff(s, other)})")
    # inline-asm IEEE ops of the kernels on denormals (Triton's libdevice add_rn / div_rn flush them to zero)
    a = torch.tensor([1e-40, -3e-39, 2.0, 1e-38, 5e-45], device=DEV)
    b = torch.tensor([3.0, 7.0, 3e38, 0.5, 1.0], device=DEV)
    out = torch.empty(3, 5, device=DEV)
    _ieee_ops_kernel[(1,)](a, b, out, 5, BLOCK=8)
    ref = torch.stack([a + b, a * b, a / b])
    check("(e) kernel IEEE add / mul / div keep denormals like eager", bits_equal(out, ref),
          f"({ndiff(out, ref)} of {ref.numel()} differ)")
    total_cases, bad = 0, []
    for n in (T, 1, 2, 33, 1000, 8192):
        for kind in ("normal", "ties", "wide", "zero-grad"):
            logits = torch.randn(n, E, generator=gen, device=DEV) * 2
            if kind == "ties":    # quantized logits: many equal scores, also at the top-k boundary
                logits = torch.randint(-3, 4, (n, E), generator=gen, device=DEV).float() * 0.5
            if kind == "wide":
                logits = torch.randn(n, E, generator=gen, device=DEV) * 30
            upstream = torch.randn(n, TOPK, generator=gen, device=DEV)
            if kind == "zero-grad":   # +0 and -0 upstream gradients
                upstream = torch.where(torch.rand(n, TOPK, generator=gen, device=DEV) < 0.5, -0.0, 0.0) * upstream.sign()
            for norm, scale, out_dtype, activation in ((True, 4.0, torch.bfloat16, "sigmoid"),
                                                      (True, 1.0, torch.bfloat16, "sigmoid"),
                                                      (False, 4.0, torch.float32, "sigmoid"),
                                                      (True, 2.5, torch.float32, "softmax")):
                bias = torch.zeros(E, device=DEV) if kind != "wide" else torch.randn(E, generator=gen, device=DEV) * 1e-3
                ref = run_router(logits, upstream, False, norm, scale, out_dtype, activation, bias)
                got = run_router(logits, upstream, True, norm, scale, out_dtype, activation, bias)
                total_cases += 1
                errs = []
                if not bits_equal(got[0], ref[0]):
                    errs.append(f"routing weights {ndiff(got[0], ref[0])}")
                if not bits_equal(got[1], ref[1]):
                    errs.append("selected experts")
                if not bits_equal(got[2], ref[2]):
                    errs.append(f"logits grad {ndiff(got[2], ref[2])} of {got[2].numel()}")
                if not bits_equal(got[3].tokens_per_expert, ref[3].tokens_per_expert):
                    errs.append("tokens_per_expert")
                a_got, a_ref = got[3].avg_topk_sigmoid_score.item(), ref[3].avg_topk_sigmoid_score.item()
                if abs(a_got - a_ref) > 1e-6 * abs(a_ref):
                    errs.append(f"avg_topk_sigmoid_score {a_got} vs {a_ref}")
                if errs:
                    bad.append(f"n={n} {kind} norm={norm} scale={scale} {out_dtype} {activation}: {', '.join(errs)}")
    for b in bad[:20]:
        print("   " + b)
    check("(e) K2c router_tail: routing weights, logits gradient, expert counts bitwise; avg score <= 1e-6",
          not bad, f"({total_cases} cases: 6 sizes x normal / ties / wide / signed-zero gradients x 4 configurations)")
    block = fake_block()
    sc = torch.rand(moe_glue._ROUTER_MAX_ROWS + 1, E, device=DEV)
    check("(e) K2c router_tail declines more rows than one CTA takes (eager fallback)",
          moe_glue.router_tail(block, sc, torch.topk(sc, TOPK, -1).indices, torch.bfloat16) is None)


# ------------------------------------------------------------------------------------------------ (d) full MoE call
def moe_call(glue, hidden, rw, sel, weights, upstream):
    module = types.SimpleNamespace(_moe_glue=glue) if glue is not None else types.SimpleNamespace()
    h = hidden.detach().clone().requires_grad_(True)
    r = rw.detach().clone().requires_grad_(True)
    ws = [w.detach().clone().requires_grad_(True) for w in weights]
    out = fused_moe_forward(module=module, num_experts=E, routing_weights=r, selected_experts=sel, hidden_states=h,
                            fc1_1_weight=ws[0], fc1_2_weight=ws[1], fc2_weight=ws[2])
    # the weights are leaves in fp32, as FSDP2 hands them over with enable_fp32 (the eager path casts them)
    out.backward(upstream)
    return out.detach(), h.grad, r.grad, [w.grad for w in ws]


def test_full_moe():
    gen = torch.Generator(device=DEV).manual_seed(4)
    hidden = torch.randn(T, HID, generator=gen, device=DEV).to(torch.bfloat16)
    sel = random_routing(T, 2.0, gen, (9,))
    rw = (torch.rand(T, TOPK, generator=gen, device=DEV) + 0.1).to(torch.bfloat16)
    weights = [torch.randn(E, FFN, HID, generator=gen, device=DEV) * 0.02,
               torch.randn(E, FFN, HID, generator=gen, device=DEV) * 0.02,
               torch.randn(E, HID, FFN, generator=gen, device=DEV) * 0.02]
    upstream = (torch.randn(T, HID, generator=gen, device=DEV) * 1e-2).to(torch.bfloat16)
    ref = moe_call(None, hidden, rw, sel, weights, upstream)
    for glue in (moe_glue.MoeGlue(True, True, True), moe_glue.MoeGlue(False, True, False), moe_glue.MoeGlue(True, False, False)):
        got = moe_call(glue, hidden, rw, sel, weights, upstream)
        same = [bits_equal(got[0], ref[0]), bits_equal(got[1], ref[1]), bits_equal(got[2], ref[2])]
        wrel = [((g.double() - r.double()).abs().max() / r.double().abs().max()).item() for g, r in zip(got[3], ref[3])]
        wsame = all(bits_equal(g, r) for g, r in zip(got[3], ref[3]))
        ok = all(same) and (wsame if not glue.fp32_weights else max(wrel) <= 1e-2)
        check(f"(d) fused_moe_forward {glue}: output / hidden grad / gate grad bitwise, weight grads "
              f"{'bitwise' if not glue.fp32_weights else '<= 1e-2'}", ok,
              f"(bitwise {same}, weight grads {'bitwise' if wsame else 'max rel ' + ', '.join(f'{x:.1e}' for x in wrel)}, "
              f"weight grad dtypes {[str(g.dtype).replace('torch.', '') for g in got[3]]})")


# ------------------------------------------------------------------------------------------------ timings
def bench(fn, n=50, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    return statistics.median(times)


def timings():
    gen = torch.Generator(device=DEV).manual_seed(7)
    print("\ntimings (median of 50, us; small kernels back to back, so mostly launch-bound like in training):")
    idx = random_routing(T, 2.0, gen)
    gate = torch.rand(T, TOPK, generator=gen, device=DEV).to(torch.bfloat16)

    def eager_index():
        splits = expert_histogram(idx, E)
        si = idx.flatten().argsort(stable=True).argsort().int().view(idx.shape)
        cs = torch.cumsum(splits, dim=0)
        sg = torch.empty_like(gate.reshape(-1, 1))
        sg[si.flatten()] = gate.reshape(-1, 1)
        return si, cs, sg
    print(f"  K2b histogram + 2 argsorts + cumsum + index_put : {bench(eager_index):7.1f}")
    print(f"  K2b route_index (1 kernel)                     : {bench(lambda: moe_glue.route_index(idx, gate, E)):7.1f}")

    logits = torch.randn(T, E, generator=gen, device=DEV, requires_grad=True)
    upstream = torch.randn(T, TOPK, generator=gen, device=DEV).to(torch.bfloat16)
    bias = torch.zeros(E, device=DEV)
    block = fake_block()
    for use in (False, True):
        def router():
            scores = logits.sigmoid()
            _, sel = torch.topk(scores + bias.unsqueeze(0), TOPK, dim=-1)
            rw = moe_glue.router_tail(block, scores, sel, torch.bfloat16) if use else eager_tail(block, scores, sel, torch.bfloat16)
            rw.backward(upstream)
        print(f"  K2c router sigmoid..cast fwd+bwd, {'Triton tail' if use else 'eager      '}   : {bench(router):7.1f}")

    sets = []
    for _ in range(3):
        cumsum, m = make_cumsum(gen)
        sets.append((cumsum, m, (torch.randn(m, HID, generator=gen, device=DEV)).to(torch.bfloat16),
                     torch.randn(E, FFN, HID, generator=gen, device=DEV) * 0.02))
    k = [0]

    def nxt():
        k[0] = (k[0] + 1) % len(sets)
        return sets[k[0]]

    def cast_gemm():
        cumsum, m, a, w = nxt()
        group_gemm_same_nk(a=a, b=w.to(torch.bfloat16), cumsum_M=cumsum, max_M=m, transpose_b=True)

    def gemm_bf16():
        cumsum, m, a, w = nxt()
        group_gemm_same_nk(a=a, b=w16[k[0]], cumsum_M=cumsum, max_M=m, transpose_b=True)

    def gemm_fp32():
        cumsum, m, a, w = nxt()
        group_gemm_same_nk(a=a, b=w, cumsum_M=cumsum, max_M=m, transpose_b=True)
    w16 = [s[3].to(torch.bfloat16) for s in sets]
    print(f"  K2a fc1 forward: weight cast + GEMM (bf16 B)     : {bench(cast_gemm):7.1f}")
    print(f"  K2a fc1 forward: GEMM alone (bf16 B)             : {bench(gemm_bf16):7.1f}")
    print(f"  K2a fc1 forward: GEMM, fp32 B cast in the kernel : {bench(gemm_fp32):7.1f}")

    hidden = torch.randn(T, HID, generator=gen, device=DEV).to(torch.bfloat16)
    sel = random_routing(T, 2.0, gen)
    rw = (torch.rand(T, TOPK, generator=gen, device=DEV) + 0.1).to(torch.bfloat16)
    wsets = [[torch.randn(E, FFN, HID, generator=gen, device=DEV) * 0.02, torch.randn(E, FFN, HID, generator=gen, device=DEV) * 0.02,
              torch.randn(E, HID, FFN, generator=gen, device=DEV) * 0.02] for _ in range(3)]
    up = (torch.randn(T, HID, generator=gen, device=DEV) * 1e-2).to(torch.bfloat16)
    for ws in wsets:
        for w in ws:
            w.requires_grad_(True)
    h = hidden.clone().requires_grad_(True)
    r = rw.clone().requires_grad_(True)
    for glue in (None, moe_glue.MoeGlue(True, True, False), moe_glue.MoeGlue(False, True, False), moe_glue.MoeGlue(True, False, False)):
        module = types.SimpleNamespace(_moe_glue=glue) if glue is not None else types.SimpleNamespace()

        def layer():
            k[0] = (k[0] + 1) % 3
            ws = wsets[k[0]]
            out = fused_moe_forward(module=module, num_experts=E, routing_weights=r, selected_experts=sel, hidden_states=h,
                                    fc1_1_weight=ws[0], fc1_2_weight=ws[1], fc2_weight=ws[2])
            out.backward(up)
            h.grad = r.grad = None
            for w in ws:
                w.grad = None
        print(f"  fused MoE fwd+bwd (no router), glue {glue}: {bench(layer, n=20):7.1f}")


def main():
    torch.manual_seed(0)
    print(f"device {torch.cuda.get_device_name()}, torch {torch.__version__}")
    test_route_index()
    test_fp32_b_gemm()
    test_fp32_wgrad()
    test_full_moe()
    test_router_tail()
    if "--no-bench" not in sys.argv:
        timings()
    print(f"\n{'ALL PASS' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
