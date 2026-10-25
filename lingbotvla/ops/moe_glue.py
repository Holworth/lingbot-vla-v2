"""MoE glue for --train.moe_glue_opt: fewer, cheaper small kernels around the fused MoE experts.

    from lingbotvla.ops.moe_glue import apply_moe_glue
    glue, n = apply_moe_glue(model)   # sets ._moe_glue on every Qwen2TokenMoeBlock

Three sub-items, each on under the switch and off with its environment variable (for attribution):

K2a  LINGBOT_MOE_FP32_WEIGHTS=0 disables it. The fp32 expert weights go into the grouped GEMMs as they are:
     group_gemm_same_nk casts the B tile to bf16 right after loading it (the same RTNE conversion as .to(bfloat16), so
     the forward and dgrad results are bitwise unchanged), and the wgrad GEMMs write their fp32 accumulators into fp32
     gradients instead of rounding them to bf16 first (more accurate, not bitwise). This drops the three weight casts
     per call (forward and recompute) and the three bf16 -> fp32 gradient casts in backward.
K2b  LINGBOT_MOE_COUNTING_SORT=0 disables it. One single-CTA Triton kernel (route_index) replaces expert_histogram,
     the two cub radix sorts of argsort(stable=True).argsort(), the cumsum and the index_put of the gate weights: a
     stable counting sort gives every routed token its row. scatter_index, cumsum_M and the scattered gate weights are
     bitwise the same.
K2c  LINGBOT_MOE_COMPILE_ROUTER=0 disables it. The routing tail after top-k (gather, the runtime stats of
     _update_moe_runtime_stats, top-k normalization, routed scaling, cast) is one Triton kernel forward and one
     backward (router_tail). Every rounding of the eager chain and of its autograd backward is reproduced: IEEE
     add / multiply / divide as explicit PTX (no FMA contraction, no flush-to-zero), the eager reduction order of the
     4-element row sums. Routing weights and gradients are therefore bitwise unchanged. sigmoid / softmax, the bias add and
     torch.topk stay eager, so the selected experts are the eager ones by construction. Of the runtime stats the
     expert counts are exact; avg_topk_sigmoid_score (monitoring only) is summed in another order.

Debug environment variables (off by default, they only add work when set):
  LINGBOT_MOE_GLUE_CHECK=1           route_index (K2b: all three outputs) and the router_tail forward (K2c: routing
                                     weights only) are recomputed with the eager code on the same inputs and the
                                     differing elements counted. K2a, the router backward and the runtime stats are
                                     not checked. Counts are per rank; the trainer prints rank 0's after each step
                                     (check_report() syncs).
  LINGBOT_MOE_ROUTING_DUMP_STEPS=1,2,3  every MoE call of these steps records router_logits, selected_experts,
                                     routing_weights and scatter_index, glue or not; routing_dump_end() writes them.
"""
from __future__ import annotations

import collections
import dataclasses
import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:
    # IEEE fp32 add / multiply / divide that the compiler can neither contract into an FMA nor flush to zero: explicit
    # PTX with a rounding modifier and without .ftz, i.e. what the eager ATen kernels compute (libdevice's add_rn etc.
    # are built flush-to-zero under Triton, which turns denormal results, e.g. small softmax probabilities, into 0).
    @triton.jit
    def _add(a, b):
        return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _mul(a, b):
        return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _div(a, b):
        return tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) != "0"


@dataclasses.dataclass(frozen=True)
class MoeGlue:
    fp32_weights: bool = True    # K2a
    counting_sort: bool = True   # K2b
    router: bool = True          # K2c

    @classmethod
    def from_env(cls) -> "MoeGlue":
        return cls(
            fp32_weights=_env_flag("LINGBOT_MOE_FP32_WEIGHTS", "1"),
            counting_sort=_env_flag("LINGBOT_MOE_COUNTING_SORT", "1"),
            router=_env_flag("LINGBOT_MOE_COMPILE_ROUTER", "1"),
        )


def apply_moe_glue(model, glue: MoeGlue = None):
    """--train.moe_glue_opt: attach the options to every Qwen2TokenMoeBlock (read by its forward and fused_moe_forward).
    Returns (options, number of blocks)."""
    from lingbotvla.distributed.parallel_state import get_parallel_state
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import Qwen2TokenMoeBlock

    if triton is None:
        raise RuntimeError("moe_glue_opt needs Triton")
    if get_parallel_state().ep_enabled:
        raise ValueError("moe_glue_opt covers the non-EP fused MoE path only")
    glue = glue or MoeGlue.from_env()
    blocks = [m for m in model.modules() if isinstance(m, Qwen2TokenMoeBlock)]
    if not blocks:
        raise ValueError("moe_glue_opt: no Qwen2TokenMoeBlock in the model")
    for block in blocks:
        if block._moe_implementation != "fused":
            raise ValueError(f"moe_glue_opt needs moe_implementation=fused, got {block._moe_implementation}")
        if glue.router and block.top_k != _ROUTER_TOPK:
            raise ValueError(f"moe_glue_opt router tail is written for top_k={_ROUTER_TOPK}, got {block.top_k}")
        block._moe_glue = glue
    return glue, len(blocks)


# ---------------------------------------------------------------------------------------------------------------
# K2b: stable counting sort of the routed tokens (one CTA)
# ---------------------------------------------------------------------------------------------------------------
_ROUTE_INDEX_MAX = 16384     # routed tokens (T * top_k) one CTA handles; larger inputs keep the eager code

if triton is not None:

    @triton.jit
    def _route_index_kernel(idx_ptr, gate_ptr, pos_ptr, cumsum_ptr, sgate_ptr, n,
                            E: tl.constexpr, BLOCK: tl.constexpr):
        # Flat token t (row-major over [T, top_k]) with expert e goes to row  start[e] + #{t' < t : e(t') = e},
        # start[e] = #{t' : e(t') < e}: the inverse permutation of argsort(stable=True), i.e. argsort().argsort().
        offs = tl.arange(0, BLOCK)
        m = offs < n
        ids = tl.load(idx_ptr + offs, mask=m, other=E).to(tl.int32)
        # active only with TRITON_DEBUG=1; masked lanes hold the sentinel E
        tl.device_assert(((ids >= 0) & (ids < E)) | (offs >= n), "expert index out of range")
        pos = tl.zeros([BLOCK], dtype=tl.int32)
        start = 0
        for e in tl.static_range(E):
            hit = ids == e
            h = hit.to(tl.int32)
            pos = tl.where(hit, start + tl.cumsum(h, axis=0) - 1, pos)
            start = start + tl.sum(h, axis=0)
            tl.store(cumsum_ptr + e, start.to(tl.int64))       # torch.cumsum(expert_histogram(...)): int64
        tl.store(pos_ptr + offs, pos, mask=m)
        tl.store(sgate_ptr + pos, tl.load(gate_ptr + offs, mask=m), mask=m)


def route_index(expert_index: torch.Tensor, gate_weights: torch.Tensor, num_experts: int):
    """For FusedMoeExpertFunction.forward: (scatter_index int32 [T, k], cumsum_M int64 [E], scattered_gate_weight
    [T*k, 1]), bitwise equal to argsort(stable=True).argsort().int(), torch.cumsum(expert_histogram(...)) and
    `scattered_gate_weight[scatter_index.flatten()] = gate_weights.reshape(-1, 1)`. None if T*k is too large."""
    n = expert_index.numel()
    if n > _ROUTE_INDEX_MAX or n == 0:
        return None
    assert expert_index.is_cuda and expert_index.dtype in (torch.int32, torch.int64)
    assert gate_weights.shape == expert_index.shape
    idx = expert_index.contiguous()
    gate = gate_weights.contiguous()
    scatter_index = torch.empty(expert_index.shape, dtype=torch.int32, device=idx.device)
    cumsum = torch.empty(num_experts, dtype=torch.int64, device=idx.device)
    sgate = torch.empty((n, 1), dtype=gate.dtype, device=idx.device)
    block = max(128, triton.next_power_of_2(n))
    with torch.cuda.device(idx.device):
        _route_index_kernel[(1,)](idx, gate, scatter_index, cumsum, sgate, n, E=num_experts, BLOCK=block,
                                  num_warps=8 if block <= 8192 else 16)
    return scatter_index, cumsum, sgate


# ---------------------------------------------------------------------------------------------------------------
# K2c: routing tail after top-k
# ---------------------------------------------------------------------------------------------------------------
_ROUTER_TOPK = 4             # the kernels spell out the 4 columns (the eager row sum is the tree (w0 + w1) + (w2 + w3))
_ROUTER_MAX_ROWS = 8192      # tokens one CTA handles in forward (the runtime stats need all rows)

if triton is not None:

    @triton.jit
    def _router_tail_fwd_kernel(s_ptr, sel_ptr, out_ptr, den_ptr, tpe_ptr, avg_ptr, N, scale,
                                E: tl.constexpr, NORM: tl.constexpr, SCALE: tl.constexpr, STATS: tl.constexpr,
                                BLOCK_R: tl.constexpr):
        # routing_weights = scores.gather(1, sel); [stats]; w / (w.sum(-1, keepdim=True) + 1e-20); * scale; .to(out)
        r = tl.arange(0, BLOCK_R)
        m = r < N
        e0 = tl.load(sel_ptr + r * 4 + 0, mask=m, other=0)
        e1 = tl.load(sel_ptr + r * 4 + 1, mask=m, other=0)
        e2 = tl.load(sel_ptr + r * 4 + 2, mask=m, other=0)
        e3 = tl.load(sel_ptr + r * 4 + 3, mask=m, other=0)
        w0 = tl.load(s_ptr + r * E + e0, mask=m, other=0.0)
        w1 = tl.load(s_ptr + r * E + e1, mask=m, other=0.0)
        w2 = tl.load(s_ptr + r * E + e2, mask=m, other=0.0)
        w3 = tl.load(s_ptr + r * E + e3, mask=m, other=0.0)
        if STATS:
            # _update_moe_runtime_stats: avg_topk_sigmoid_score = mean(w) (any summation order: monitoring only);
            # tokens_per_expert += one_hot(sel).sum(0) (integer counts, exact)
            tot = tl.sum((w0 + w1) + (w2 + w3), axis=0)
            tl.store(avg_ptr, tot / (N * 4))
            cnt = (tl.histogram(e0.to(tl.int32), E, mask=m) + tl.histogram(e1.to(tl.int32), E, mask=m)
                   + tl.histogram(e2.to(tl.int32), E, mask=m) + tl.histogram(e3.to(tl.int32), E, mask=m))
            eo = tl.arange(0, E)
            tl.store(tpe_ptr + eo, tl.load(tpe_ptr + eo) + cnt.to(tl.float32))
        zero = tl.zeros([BLOCK_R], dtype=tl.float32)   # the inline-asm ops take tensors: constants as blocks
        if NORM:
            s = _add(_add(w0, w1), _add(w2, w3))
            den = _add(s, zero + 1e-20)
            tl.store(den_ptr + r, den, mask=m)
            w0 = _div(w0, den)
            w1 = _div(w1, den)
            w2 = _div(w2, den)
            w3 = _div(w3, den)
        if SCALE:
            sc = zero + scale
            w0 = _mul(w0, sc)
            w1 = _mul(w1, sc)
            w2 = _mul(w2, sc)
            w3 = _mul(w3, sc)
        ty = out_ptr.dtype.element_ty
        tl.store(out_ptr + r * 4 + 0, w0.to(ty), mask=m)
        tl.store(out_ptr + r * 4 + 1, w1.to(ty), mask=m)
        tl.store(out_ptr + r * 4 + 2, w2.to(ty), mask=m)
        tl.store(out_ptr + r * 4 + 3, w3.to(ty), mask=m)

    @triton.jit
    def _router_tail_bwd_kernel(go_ptr, s_ptr, sel_ptr, den_ptr, gs_ptr, N, scale,
                                E: tl.constexpr, NORM: tl.constexpr, SCALE: tl.constexpr, BLOCK_R: tl.constexpr):
        # The eager autograd chain, step by step: ToCopyBackward (exact), MulBackward0 g * scale,
        # DivBackward0 self: g / den, other: sum_to((-g) * ((w / den) / den)) (tree order of the 4-element sum),
        # AddBackward (1e-20): identity, SumBackward1: expand, accumulation g_self + g_den,
        # GatherBackward0: zeros.scatter_add_(1, sel, g) (0 + g turns -0 into +0).
        r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
        m = r < N
        g0 = tl.load(go_ptr + r * 4 + 0, mask=m, other=0.0).to(tl.float32)
        g1 = tl.load(go_ptr + r * 4 + 1, mask=m, other=0.0).to(tl.float32)
        g2 = tl.load(go_ptr + r * 4 + 2, mask=m, other=0.0).to(tl.float32)
        g3 = tl.load(go_ptr + r * 4 + 3, mask=m, other=0.0).to(tl.float32)
        e0 = tl.load(sel_ptr + r * 4 + 0, mask=m, other=0)
        e1 = tl.load(sel_ptr + r * 4 + 1, mask=m, other=0)
        e2 = tl.load(sel_ptr + r * 4 + 2, mask=m, other=0)
        e3 = tl.load(sel_ptr + r * 4 + 3, mask=m, other=0)
        zero = tl.zeros([BLOCK_R], dtype=tl.float32)   # the inline-asm ops take tensors: constants as blocks
        if SCALE:
            sc = zero + scale
            g0 = _mul(g0, sc)
            g1 = _mul(g1, sc)
            g2 = _mul(g2, sc)
            g3 = _mul(g3, sc)
        if NORM:
            den = tl.load(den_ptr + r, mask=m, other=1.0)
            w0 = tl.load(s_ptr + r * E + e0, mask=m, other=0.0)
            w1 = tl.load(s_ptr + r * E + e1, mask=m, other=0.0)
            w2 = tl.load(s_ptr + r * E + e2, mask=m, other=0.0)
            w3 = tl.load(s_ptr + r * E + e3, mask=m, other=0.0)
            t0 = _mul(-g0, _div(_div(w0, den), den))
            t1 = _mul(-g1, _div(_div(w1, den), den))
            t2 = _mul(-g2, _div(_div(w2, den), den))
            t3 = _mul(-g3, _div(_div(w3, den), den))
            gd = _add(_add(t0, t1), _add(t2, t3))
            g0 = _add(_div(g0, den), gd)
            g1 = _add(_div(g1, den), gd)
            g2 = _add(_div(g2, den), gd)
            g3 = _add(_div(g3, den), gd)
        g0 = _add(g0, zero)
        g1 = _add(g1, zero)
        g2 = _add(g2, zero)
        g3 = _add(g3, zero)
        c = tl.arange(0, E)[None, :]
        tile = tl.where(c == e0[:, None], g0[:, None],
                        tl.where(c == e1[:, None], g1[:, None],
                                 tl.where(c == e2[:, None], g2[:, None],
                                          tl.where(c == e3[:, None], g3[:, None], 0.0))))
        tl.store(gs_ptr + r[:, None] * E + c, tile, mask=m[:, None])


class _RouterTail(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, sel, norm, scale, out_dtype, tpe, avg):
        n_rows, num_experts = scores.shape
        out = torch.empty((n_rows, _ROUTER_TOPK), dtype=out_dtype, device=scores.device)
        den = torch.empty((n_rows,), dtype=torch.float32, device=scores.device) if norm else scores
        stats = tpe is not None
        with torch.cuda.device(scores.device):
            _router_tail_fwd_kernel[(1,)](
                scores, sel, out, den, tpe if stats else scores, avg if stats else scores, n_rows, scale,
                E=num_experts, NORM=norm, SCALE=scale != 1.0, STATS=stats,
                BLOCK_R=max(128, triton.next_power_of_2(n_rows)), num_warps=8,
            )
        ctx.save_for_backward(scores, sel, den)
        ctx.norm, ctx.scale = norm, scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        scores, sel, den = ctx.saved_tensors
        n_rows, num_experts = scores.shape
        grad_scores = torch.empty_like(scores)
        block_r = 64
        with torch.cuda.device(scores.device):
            _router_tail_bwd_kernel[(triton.cdiv(n_rows, block_r),)](
                grad_out.contiguous(), scores, sel, den, grad_scores, n_rows, ctx.scale,
                E=num_experts, NORM=ctx.norm, SCALE=ctx.scale != 1.0, BLOCK_R=block_r, num_warps=4,
            )
        return grad_scores, None, None, None, None, None, None


def router_tail(block, routing_scores: torch.Tensor, selected_experts: torch.Tensor, out_dtype: torch.dtype):
    """Qwen2TokenMoeBlock.forward from `routing_scores.gather(1, selected_experts)` to `.to(hidden_states.dtype)`,
    including _update_moe_runtime_stats in training. None when the shape is not covered (the caller runs eager)."""
    n_rows, num_experts = routing_scores.shape
    if (selected_experts.shape != (n_rows, _ROUTER_TOPK) or n_rows > _ROUTER_MAX_ROWS or n_rows == 0
            or not routing_scores.is_cuda or routing_scores.dtype != torch.float32 or num_experts > 1024
            or num_experts & (num_experts - 1)):
        return None
    stats = block.training and hasattr(block, "tokens_per_expert") and hasattr(block, "avg_topk_sigmoid_score")
    scale = float(block.routed_scaling_factor)
    out = _RouterTail.apply(
        routing_scores.contiguous(), selected_experts.contiguous(), bool(block.norm_topk_prob), scale, out_dtype,
        block.tokens_per_expert if stats else None, block.avg_topk_sigmoid_score if stats else None,
    )
    if CHECK:
        _check_router(block, routing_scores, selected_experts, out_dtype, out)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Debug: LINGBOT_MOE_GLUE_CHECK (glue vs eager on the same inputs) and LINGBOT_MOE_ROUTING_DUMP_STEPS
# ---------------------------------------------------------------------------------------------------------------
CHECK = os.environ.get("LINGBOT_MOE_GLUE_CHECK", "0") == "1"
_check_counts = collections.OrderedDict()   # name -> [calls, differing elements (GPU tensor)]


def _count(name, a, b):
    entry = _check_counts.setdefault(name, [0, None])
    diff = (a.reshape(-1) != b.reshape(-1)).sum() if a.shape == b.shape else torch.tensor(a.numel() + 1, device=a.device)
    entry[0] += 1
    entry[1] = diff if entry[1] is None else entry[1] + diff


@torch.no_grad()
def _check_router(block, routing_scores, selected_experts, out_dtype, out):
    ref = routing_scores.gather(1, selected_experts)
    if block.norm_topk_prob:
        ref = ref / (ref.sum(dim=-1, keepdim=True) + 1e-20)
    if block.routed_scaling_factor != 1.0:
        ref = ref * block.routed_scaling_factor
    _count("router_tail/routing_weights", out, ref.to(out_dtype))


@torch.no_grad()
def check_route_index(expert_index, gate_weights, num_experts, scatter_index, cumsum, sgate):
    from lingbotvla.ops.group_gemm.kernel.moe import expert_histogram

    ref_index = expert_index.flatten().argsort(stable=True).argsort().int().view(expert_index.shape)
    ref_cumsum = torch.cumsum(expert_histogram(expert_index, num_experts), dim=0)
    ref_sgate = torch.empty_like(gate_weights.reshape(-1, 1))
    ref_sgate[ref_index.flatten()] = gate_weights.reshape(-1, 1)
    _count("route_index/scatter_index", scatter_index, ref_index)
    _count("route_index/cumsum_M", cumsum, ref_cumsum)
    _count("route_index/scattered_gate_weight", sgate, ref_sgate)


def check_report() -> str:
    """Totals since the last call ('name: calls, differing elements'); syncs. Empty string when nothing was checked."""
    if not _check_counts:
        return ""
    parts = [f"{name}: {calls} calls, {int(diff.item())} differing" for name, (calls, diff) in _check_counts.items()]
    _check_counts.clear()
    return "; ".join(parts)


def _parse_steps(value: str):
    return {int(s) for s in value.replace(" ", "").split(",") if s}


DUMP_STEPS = _parse_steps(os.environ.get("LINGBOT_MOE_ROUTING_DUMP_STEPS", ""))
DUMP = False                 # True while a dump step runs (set by routing_dump_begin)
_dump_records = []


def routing_dump_begin(step: int) -> None:
    global DUMP
    DUMP = step in DUMP_STEPS
    _dump_records.clear()


def routing_dump_record(name: str, tensor: torch.Tensor) -> None:
    _dump_records.append((name, tensor.detach().clone()))


def routing_dump_end(step: int, rank: int, out_dir: str) -> None:
    global DUMP
    if DUMP and _dump_records:
        path = os.path.join(out_dir, "routing_dump")
        os.makedirs(path, exist_ok=True)
        torch.save([(name, t.cpu()) for name, t in _dump_records], os.path.join(path, f"step{step}_rank{rank}.pt"))
    _dump_records.clear()
    DUMP = False
