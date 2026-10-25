"""Correctness and timing tests for lingbotvla/ops/swiglu.py.

    python profile/bench/test_swiglu.py            # CPU + GPU (if available)
    TRITON_INTERPRET=1 python profile/bench/test_swiglu.py --interpret                   # Triton kernels on the CPU interpreter

Compares `swiglu(gate, up)` (exact mode) with the eager chain `F.silu(gate) * up`, forward and backward, and reports
how many elements differ and by how many bf16 ulps.  PASS criteria: in bf16 (the training
dtype) exact mode must match eager bit-for-bit on CPU and may differ by at most 1 bf16 ulp on < 0.1% of elements on
GPU (FMA contraction may differ between nvcc and Triton); in fp32 the d_gate formula is evaluated in a different
association than ATen's vectorized CPU kernel, so 1 fp32 ulp (2e-6 abs on O(1) values) is allowed.  Also checks
`apply_fused_swiglu` on a toy HF-like MLP module and times eager vs fused at the training shape [9152, 9728].
NaN/Inf in the inputs and in the upstream gradient must come out where eager puts them (bf16).
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from lingbotvla.ops.swiglu import apply_fused_swiglu, swiglu, swiglu_reference  # noqa: E402

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
    from lingbotvla.ops.swiglu import _rtne_bf16_bits as _rtne_helper  # noqa: E402
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


def ulp_report(a, b):
    """bf16 ulp distance between two tensors of the same dtype (via int16 bit patterns)."""
    if a.dtype == torch.bfloat16:
        d = (a.view(torch.int16).int() - b.view(torch.int16).int()).abs()
        return d.max().item(), (d > 0).float().mean().item()
    d = (a.float() - b.float()).abs()
    return d.max().item(), (d > 0).float().mean().item()


def run_equivalence(device, shape, dtype, tag, max_ulp, max_frac):
    gen = torch.Generator(device="cpu").manual_seed(0)
    g = (torch.randn(*shape, generator=gen) * 3).to(device=device, dtype=dtype).requires_grad_(True)
    u = torch.randn(*shape, generator=gen).to(device=device, dtype=dtype).requires_grad_(True)
    do = torch.randn(*shape, generator=gen).to(device=device, dtype=dtype)
    y_ref = swiglu_reference(g, u)
    gg_ref, gu_ref = torch.autograd.grad((y_ref * do).sum(), (g, u))
    y = swiglu(g, u)
    gg, gu = torch.autograd.grad((y * do).sum(), (g, u))
    for name, a, b in (("forward", y, y_ref), ("d_gate", gg, gg_ref), ("d_up", gu, gu_ref)):
        mu, fr = ulp_report(a.detach(), b.detach())
        check(f"{tag} {name}", mu <= max_ulp and fr <= max_frac, f"max_ulp={mu} mismatch_frac={fr:.2e}")


def run_apply():
    class Cfg:
        hidden_act = "silu"

    class Qwen3VLTextMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Cfg()
            self.gate_proj = torch.nn.Linear(16, 40, bias=False)
            self.up_proj = torch.nn.Linear(16, 40, bias=False)
            self.down_proj = torch.nn.Linear(40, 16, bias=False)
            self.act_fn = F.silu

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    m = Qwen3VLTextMLP()
    x = torch.randn(3, 5, 16)
    ref = m(x)
    n = apply_fused_swiglu(torch.nn.Sequential(m, torch.nn.Linear(16, 16)))
    out = m(x)
    check("apply_fused_swiglu patches 1 module", n == 1, f"n={n}")
    check("patched MLP output equals original (fp32)", torch.allclose(out, ref, rtol=1e-6, atol=1e-6))


def run_nonfinite(device, tag, max_ulp):
    """NaN/Inf in gate, up and the upstream gradient (bf16): NaN/Inf where eager has them, finite values as usual."""
    gen = torch.Generator(device="cpu").manual_seed(2)
    g = torch.randn(37, 300, generator=gen) * 3
    u = torch.randn(37, 300, generator=gen)
    do = torch.randn(37, 300, generator=gen)
    g[0, 0], g[1, 1], g[2, 2] = float("nan"), float("inf"), -float("inf")
    u[3, 3], u[4, 4] = float("nan"), float("inf")
    do[5, 5], do[6, 6] = float("nan"), -float("inf")
    g, u = (t.to(device=device, dtype=torch.bfloat16).requires_grad_(True) for t in (g, u))
    do = do.to(device=device, dtype=torch.bfloat16)
    y_ref = swiglu_reference(g, u)
    gg_ref, gu_ref = torch.autograd.grad((y_ref * do).sum(), (g, u))
    y = swiglu(g, u)
    gg, gu = torch.autograd.grad((y * do).sum(), (g, u))
    ok, n = True, 0
    for a, b in ((y, y_ref), (gg, gg_ref), (gu, gu_ref)):
        a, b = a.detach(), b.detach()
        nan, inf = torch.isnan(b), torch.isinf(b)
        fin = ~(nan | inf)
        mu = (a[fin].view(torch.int16).int() - b[fin].view(torch.int16).int()).abs().max().item()
        ok = ok and torch.equal(torch.isnan(a), nan) and torch.equal(a[inf], b[inf]) and mu <= max_ulp
        n += int(nan.sum()) + int(inf.sum())
    check(f"{tag} non-finite inputs and grads", ok, f"({n} NaN/Inf output elements, all where eager has them)")


def run_timing(device, shape, dtype, iters=20):
    g = torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)
    u = torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)
    do = torch.randn(*shape, device=device, dtype=dtype)
    for name, fn in (("eager silu*mul", swiglu_reference), ("fused swiglu (exact)", swiglu),
                     ("fused swiglu (exact=False)", lambda a, b: swiglu(a, b, exact=False))):
        for _ in range(3):
            y = fn(g, u); torch.autograd.grad((y * do).sum(), (g, u))
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            s.record()
            for _ in range(iters):
                fn(g, u)
            e.record(); torch.cuda.synchronize()
        fwd = s.elapsed_time(e) / iters * 1e3
        t0 = time.perf_counter()
        for _ in range(iters):
            y = fn(g, u); torch.autograd.grad((y * do).sum(), (g, u))
        torch.cuda.synchronize()
        tot = (time.perf_counter() - t0) / iters * 1e6
        print(f"  {name:28s} forward {fwd:8.1f} us   forward+backward {tot:8.1f} us   (per layer)")


if __name__ == "__main__":
    if "--interpret" in sys.argv:
        import lingbotvla.ops.swiglu as _m
        assert os.environ.get("TRITON_INTERPRET") == "1" and _m.triton is not None, "set TRITON_INTERPRET=1 and install triton"
        _m._use_triton = lambda t: True
        run_equivalence("cpu", (37, 300), torch.bfloat16, "interpreter bf16", max_ulp=0, max_frac=0.0)
        run_equivalence("cpu", (37, 300), torch.float32, "interpreter fp32", max_ulp=2e-6, max_frac=1.0)
        run_nonfinite("cpu", "interpreter", max_ulp=0)
        run_rounding_payloads("cpu", "interpreter")
        print("FAILED:" if FAILED else "ALL PASSED (interpreter)", FAILED or "")
        sys.exit(1 if FAILED else 0)
    run_equivalence("cpu", (37, 300), torch.bfloat16, "CPU bf16", max_ulp=0, max_frac=0.0)
    run_equivalence("cpu", (37, 300), torch.float32, "CPU fp32", max_ulp=2e-6, max_frac=1.0)
    run_nonfinite("cpu", "CPU", max_ulp=0)
    run_apply()
    if torch.cuda.is_available():
        # training shape: [B*L, intermediate] = [32*286, 9728] bf16
        run_equivalence("cuda", (9152, 9728), torch.bfloat16, "GPU bf16 training shape", max_ulp=1, max_frac=1e-3)
        run_nonfinite("cuda", "GPU", max_ulp=1)
        run_rounding_payloads("cuda", "GPU")
        run_equivalence("cuda", (9152, 9728), torch.float32, "GPU fp32", max_ulp=2e-6, max_frac=1.0)
        run_timing("cuda", (9152, 9728), torch.bfloat16)
    else:
        print("CUDA not available: GPU checks skipped")
    if FAILED:
        print("FAILED:", FAILED)
        sys.exit(1)
    print("ALL PASSED")
