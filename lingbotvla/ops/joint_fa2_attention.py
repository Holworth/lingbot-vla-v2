"""Joint (VLM + action-expert) attention as two FlashAttention-2 varlen calls, replacing flex for the training mask.

The training mask (make_att_2d_masks with vlm_causal=True, suffix mask_ar = [1, 1, 0, ...]) is, per sample b:

    rows 0..Lv-1  (prefix, Lv tokens) : causal over the prefix           -> key k visible iff k <= q
    row  Lv       (state)             : all prefix + itself              -> k <= Lv
    rows Lv+1..L-1 (Na actions)       : everything (prefix, state, actions, bidirectional)
    AND pad[b, q] & pad[b, k]         : language pad tokens are invisible as keys and produce empty rows as queries

Rows 0..Lv (prefix + state) therefore form one *causal* attention over the valid prefix+state tokens, and the Na
action rows form one *full* attention over all valid tokens.  The two query sets are disjoint, so no LSE merging is
needed and each can be a plain FlashAttention-2 call.  Pad is removed by packing per sample (varlen `cu_seqlens`).

    call A : q = k = v = [valid prefix tokens, state]  per sample, causal=True
    call B : q = [Na actions], k = v = [valid prefix, state, actions] per sample, causal=False

`joint_fa2_attention(...)` = pack (one Triton kernel per q/k/v: cast + RoPE + scatter into the packed FA2 layouts,
no concat, no padding) -> 2x flash_attn_varlen_func (bf16) -> unpack (one Triton kernel: gather the two outputs back
into att_vlm [B, Lv, Hq*D] (vlm_dtype, pad rows = 0) and att_exp [B, Le, Hq*D] (exp_dtype)).  Backward mirrors it.

Buffers are allocated at static upper bounds (B*(Lv+1) rows for call A, B*L for call-B keys) so that no GPU->CPU
synchronisation is needed: FlashAttention only reads rows below cu_seqlens[-1].  `max_seqlen_*` are static bounds too.

The plan (index maps + cu_seqlens) depends only on pad_masks and is built once per forward; `plan_to_dense_mask`
reconstructs the dense [B, L, L] mask the plan implements, for checking against the model's att_2d_masks.

On CUDA with flash_attn installed the FA2 kernels are used; on CPU (unit tests) an equivalent torch varlen attention
runs so that the whole pipeline (packing, masks, GQA, backward) is testable without a GPU.
The Triton kernels are compiled with enable_fp_fusion=False (no mul+add -> FMA contraction), like joint_qkv_prep's:
the packed bf16 Q/K/V are then bit-identical to joint_qkv_prep(out_dtype=bf16)'s, and the kernels to their torch
fallbacks.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:  # Triton is optional so that the CPU path (tests) works without it
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

try:
    from flash_attn import flash_attn_varlen_func as _fa2_varlen_func
    _FA2_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    _fa2_varlen_func = None
    _FA2_IMPORT_ERROR = e

_BLOCK_T = int(os.environ.get("JOINT_FA2_BLOCK_T", "16"))
_NUM_WARPS = int(os.environ.get("JOINT_FA2_NUM_WARPS", "4"))
_FORCE_TORCH_ATTN = os.environ.get("JOINT_FA2_FORCE_TORCH", "0") == "1"


# --------------------------------------------------------------------------------------------------------------
# Plan: index maps and cu_seqlens from pad_masks
# --------------------------------------------------------------------------------------------------------------
@dataclass
class FA2Plan:
    B: int
    Lv: int
    Le: int
    L: int
    Na: int
    dstA: torch.Tensor    # [B, L] int32: row in the call-A space for valid prefix tokens and the state, else -1
    dstBk: torch.Tensor   # [B, L] int32: row in the call-B key space for every valid token, else -1
    dstBq: torch.Tensor   # [B, L] int32: row in the call-B query space for the action tokens, else -1
    cuA: torch.Tensor     # [B+1] int32
    cuBk: torch.Tensor    # [B+1] int32
    cuBq: torch.Tensor    # [B+1] int32  (= arange(B+1) * Na)
    rowsA: int            # B*(Lv+1)  upper bound of cuA[-1]
    rowsBk: int           # B*L       upper bound of cuBk[-1]
    rowsBq: int           # B*Na      (exact)
    max_a: int            # Lv+1      static bound for max_seqlen_q/k of call A
    max_bq: int           # Na
    max_bk: int           # L


def build_fa2_plan(pad_masks: torch.Tensor, Lv: int, Le: int) -> FA2Plan:
    """pad_masks: [B, L] bool (True = real token), L = Lv + Le; suffix (state + actions) must be all True."""
    assert pad_masks.dim() == 2 and pad_masks.dtype == torch.bool, (pad_masks.shape, pad_masks.dtype)
    B, L = pad_masks.shape
    assert L == Lv + Le and Le >= 2, (L, Lv, Le)
    Na = Le - 1
    dev = pad_masks.device
    pad = pad_masks
    pos = torch.arange(L, device=dev)
    i32 = torch.int32

    tok_a = pad & (pos <= Lv)[None, :]                                        # valid prefix + state
    lenA = tok_a.sum(1, dtype=i32)
    cuA = F.pad(lenA.cumsum(0, dtype=i32), (1, 0))
    dstA = torch.where(tok_a, tok_a.cumsum(1, dtype=i32) - 1 + cuA[:-1, None], -1).to(i32)

    lenB = pad.sum(1, dtype=i32)                                               # all valid tokens
    cuBk = F.pad(lenB.cumsum(0, dtype=i32), (1, 0))
    dstBk = torch.where(pad, pad.cumsum(1, dtype=i32) - 1 + cuBk[:-1, None], -1).to(i32)

    is_act = (pos > Lv)[None, :].expand(B, L)
    dstBq = torch.where(is_act, (pos - (Lv + 1))[None, :] + (torch.arange(B, device=dev) * Na)[:, None], -1).to(i32)
    cuBq = torch.arange(B + 1, device=dev, dtype=i32) * Na

    return FA2Plan(B, Lv, Le, L, Na, dstA.contiguous(), dstBk.contiguous(), dstBq.contiguous(),
                   cuA.contiguous(), cuBk.contiguous(), cuBq.contiguous(),
                   B * (Lv + 1), B * L, B * Na, Lv + 1, Na, L)


def plan_to_dense_mask(plan: FA2Plan) -> torch.Tensor:
    """The dense bool mask [B, L, L] (True = visible) implemented by the two FA2 calls of `plan`."""
    pad = plan.dstBk >= 0                                                     # [B, L]
    pos = torch.arange(plan.L, device=pad.device)
    q_pre = (pos <= plan.Lv)[:, None]                                         # prefix or state row
    k_pre = (pos <= plan.Lv)[None, :]
    causal = pos[None, :] <= pos[:, None]
    rule = torch.where(q_pre, causal & k_pre, torch.ones_like(causal))        # [L, L]
    return pad[:, :, None] & pad[:, None, :] & rule[None]


def check_plan_against_mask(plan: FA2Plan, att_2d_masks: torch.Tensor) -> None:
    """Raise if the model's dense mask is not exactly what the plan implements (debug / validation only: syncs)."""
    pad = plan.dstBk >= 0
    if not bool(pad[:, plan.Lv:].all()):
        raise RuntimeError("joint_fa2_attention: suffix (state/actions) contains pad tokens; plan assumes none")
    dense = plan_to_dense_mask(plan)
    if att_2d_masks.dtype != torch.bool:
        att_2d_masks = att_2d_masks.bool()
    if not torch.equal(dense, att_2d_masks):
        diff = (dense != att_2d_masks).sum().item()
        raise RuntimeError(f"joint_fa2_attention: plan mask differs from att_2d_masks in {diff} entries")


# --------------------------------------------------------------------------------------------------------------
# Triton kernels
# --------------------------------------------------------------------------------------------------------------
if triton is not None:

    @triton.jit
    def _rtne_bf16_bits(x):
        # round-to-nearest-even fp32 -> bf16 bit pattern (same as torch's .to(bfloat16) on every backend,
        # including TRITON_INTERPRET=1 whose implicit store cast truncates)
        u = x.to(tl.int32, bitcast=True)
        lsb = (u >> 16) & 1
        r = (u + 0x7FFF + lsb) >> 16
        # NaN stays NaN (0x7FFF, which CUDA's cvt.rn.bf16.f32, i.e. eager .to(bfloat16) on the GPU, gives); the carry
        # above would turn it into +-0 or +-inf. Integer test: Triton compiles float compares assuming no NaNs.
        r = tl.where((u & 0x7FFFFFFF) > 0x7F800000, 0x7FFF, r)
        return r.to(tl.int16)

    @triton.jit
    def _pack_fwd(
        xv_ptr, xe_ptr, cos_ptr, sin_ptr, d1_ptr, d2_ptr, o1_ptr, o2_ptr,
        Lv, L, H,
        s_vb, s_vt, s_vh,
        s_eb, s_et, s_eh,
        s_cb, s_ct,
        s_db,
        s_o1t, s_o1h, s_o2t, s_o2h,
        ROPE: tl.constexpr, HALF: tl.constexpr, BLOCK_T: tl.constexpr, OUT_BF16: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        b = pid_bh // H
        h = pid_bh - b * H
        t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
        d = tl.arange(0, HALF)
        in_l = t < L
        is_v = t < Lv
        mv = in_l & is_v
        me = in_l & (t >= Lv)
        tv = tl.where(is_v, t, 0)
        te = tl.where(is_v, 0, t - Lv)
        offv = b * s_vb + tv[:, None] * s_vt + h * s_vh + d[None, :]
        offe = b * s_eb + te[:, None] * s_et + h * s_eh + d[None, :]
        xv1 = tl.load(xv_ptr + offv, mask=mv[:, None], other=0.0).to(tl.float32)
        xv2 = tl.load(xv_ptr + offv + HALF, mask=mv[:, None], other=0.0).to(tl.float32)
        xe1 = tl.load(xe_ptr + offe, mask=me[:, None], other=0.0).to(tl.float32)
        xe2 = tl.load(xe_ptr + offe + HALF, mask=me[:, None], other=0.0).to(tl.float32)
        x1 = tl.where(is_v[:, None], xv1, xe1)
        x2 = tl.where(is_v[:, None], xv2, xe2)
        if ROPE:
            offc = b * s_cb + t[:, None] * s_ct + d[None, :]
            c = tl.load(cos_ptr + offc, mask=in_l[:, None], other=0.0).to(tl.float32)
            s = tl.load(sin_ptr + offc, mask=in_l[:, None], other=0.0).to(tl.float32)
            y1 = x1 * c - x2 * s
            y2 = x2 * c + x1 * s
        else:
            y1 = x1
            y2 = x2
        r1 = tl.load(d1_ptr + b * s_db + t, mask=in_l, other=-1)
        r2 = tl.load(d2_ptr + b * s_db + t, mask=in_l, other=-1)
        m1 = in_l & (r1 >= 0)
        m2 = in_l & (r2 >= 0)
        off1 = tl.where(m1, r1, 0)[:, None] * s_o1t + h * s_o1h + d[None, :]
        off2 = tl.where(m2, r2, 0)[:, None] * s_o2t + h * s_o2h + d[None, :]
        if OUT_BF16:
            o1i = o1_ptr.to(tl.pointer_type(tl.int16))
            o2i = o2_ptr.to(tl.pointer_type(tl.int16))
            tl.store(o1i + off1, _rtne_bf16_bits(y1), mask=m1[:, None])
            tl.store(o1i + off1 + HALF, _rtne_bf16_bits(y2), mask=m1[:, None])
            tl.store(o2i + off2, _rtne_bf16_bits(y1), mask=m2[:, None])
            tl.store(o2i + off2 + HALF, _rtne_bf16_bits(y2), mask=m2[:, None])
        else:
            tl.store(o1_ptr + off1, y1, mask=m1[:, None])
            tl.store(o1_ptr + off1 + HALF, y2, mask=m1[:, None])
            tl.store(o2_ptr + off2, y1, mask=m2[:, None])
            tl.store(o2_ptr + off2 + HALF, y2, mask=m2[:, None])

    @triton.jit
    def _pack_bwd(
        g1_ptr, g2_ptr, cos_ptr, sin_ptr, d1_ptr, d2_ptr, dv_ptr, de_ptr,
        Lv, L, H,
        s_g1t, s_g1h, s_g2t, s_g2h,
        s_cb, s_ct,
        s_db,
        s_vb, s_vt, s_vh,
        s_eb, s_et, s_eh,
        ROPE: tl.constexpr, HALF: tl.constexpr, BLOCK_T: tl.constexpr, VLM_BF16: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        b = pid_bh // H
        h = pid_bh - b * H
        t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
        d = tl.arange(0, HALF)
        in_l = t < L
        is_v = t < Lv
        mv = in_l & is_v
        me = in_l & (t >= Lv)
        r1 = tl.load(d1_ptr + b * s_db + t, mask=in_l, other=-1)
        r2 = tl.load(d2_ptr + b * s_db + t, mask=in_l, other=-1)
        m1 = in_l & (r1 >= 0)
        m2 = in_l & (r2 >= 0)
        off1 = tl.where(m1, r1, 0)[:, None] * s_g1t + h * s_g1h + d[None, :]
        off2 = tl.where(m2, r2, 0)[:, None] * s_g2t + h * s_g2h + d[None, :]
        g1a = tl.load(g1_ptr + off1, mask=m1[:, None], other=0.0).to(tl.float32)
        g1b = tl.load(g1_ptr + off1 + HALF, mask=m1[:, None], other=0.0).to(tl.float32)
        g2a = tl.load(g2_ptr + off2, mask=m2[:, None], other=0.0).to(tl.float32)
        g2b = tl.load(g2_ptr + off2 + HALF, mask=m2[:, None], other=0.0).to(tl.float32)
        ga = g1a + g2a
        gb = g1b + g2b
        if ROPE:
            offc = b * s_cb + t[:, None] * s_ct + d[None, :]
            c = tl.load(cos_ptr + offc, mask=in_l[:, None], other=0.0).to(tl.float32)
            s = tl.load(sin_ptr + offc, mask=in_l[:, None], other=0.0).to(tl.float32)
            dx1 = ga * c + gb * s
            dx2 = gb * c - ga * s
        else:
            dx1 = ga
            dx2 = gb
        tv = tl.where(is_v, t, 0)
        te = tl.where(is_v, 0, t - Lv)
        offv = b * s_vb + tv[:, None] * s_vt + h * s_vh + d[None, :]
        offe = b * s_eb + te[:, None] * s_et + h * s_eh + d[None, :]
        if VLM_BF16:
            dvi = dv_ptr.to(tl.pointer_type(tl.int16))
            tl.store(dvi + offv, _rtne_bf16_bits(dx1), mask=mv[:, None])
            tl.store(dvi + offv + HALF, _rtne_bf16_bits(dx2), mask=mv[:, None])
        else:
            tl.store(dv_ptr + offv, dx1, mask=mv[:, None])
            tl.store(dv_ptr + offv + HALF, dx2, mask=mv[:, None])
        tl.store(de_ptr + offe, dx1, mask=me[:, None])
        tl.store(de_ptr + offe + HALF, dx2, mask=me[:, None])

    @triton.jit
    def _unpack_fwd(
        o1_ptr, o2_ptr, d1_ptr, d2_ptr, v_ptr, e_ptr,
        Lv, L, H,
        s_o1t, s_o1h, s_o2t, s_o2h,
        s_db,
        s_vb, s_vt, s_eb, s_et,
        D: tl.constexpr, BLOCK_T: tl.constexpr, VLM_BF16: tl.constexpr, EXP_BF16: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        b = pid_bh // H
        h = pid_bh - b * H
        t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
        d = tl.arange(0, D)
        in_l = t < L
        is_v = t < Lv
        mv = in_l & is_v
        me = in_l & (t >= Lv)
        r1 = tl.load(d1_ptr + b * s_db + t, mask=in_l, other=-1)
        r2 = tl.load(d2_ptr + b * s_db + t, mask=in_l, other=-1)
        m1 = in_l & (r1 >= 0)
        m2 = in_l & (r2 >= 0)
        x1 = tl.load(o1_ptr + tl.where(m1, r1, 0)[:, None] * s_o1t + h * s_o1h + d[None, :], mask=m1[:, None], other=0.0).to(tl.float32)
        x2 = tl.load(o2_ptr + tl.where(m2, r2, 0)[:, None] * s_o2t + h * s_o2h + d[None, :], mask=m2[:, None], other=0.0).to(tl.float32)
        x = x1 + x2                                   # a token has at most one source row; pad rows stay 0
        tv = tl.where(is_v, t, 0)
        te = tl.where(is_v, 0, t - Lv)
        offv = b * s_vb + tv[:, None] * s_vt + h * D + d[None, :]
        offe = b * s_eb + te[:, None] * s_et + h * D + d[None, :]
        if VLM_BF16:
            tl.store(v_ptr.to(tl.pointer_type(tl.int16)) + offv, _rtne_bf16_bits(x), mask=mv[:, None])
        else:
            tl.store(v_ptr + offv, x, mask=mv[:, None])
        if EXP_BF16:
            tl.store(e_ptr.to(tl.pointer_type(tl.int16)) + offe, _rtne_bf16_bits(x), mask=me[:, None])
        else:
            tl.store(e_ptr + offe, x, mask=me[:, None])

    @triton.jit
    def _unpack_bwd(
        gv_ptr, ge_ptr, d1_ptr, d2_ptr, g1_ptr, g2_ptr,
        Lv, L, H,
        s_vb, s_vt, s_eb, s_et,
        s_db,
        s_g1t, s_g1h, s_g2t, s_g2h,
        D: tl.constexpr, BLOCK_T: tl.constexpr, OUT_BF16: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        b = pid_bh // H
        h = pid_bh - b * H
        t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
        d = tl.arange(0, D)
        in_l = t < L
        is_v = t < Lv
        mv = in_l & is_v
        me = in_l & (t >= Lv)
        tv = tl.where(is_v, t, 0)
        te = tl.where(is_v, 0, t - Lv)
        gv = tl.load(gv_ptr + b * s_vb + tv[:, None] * s_vt + h * D + d[None, :], mask=mv[:, None], other=0.0).to(tl.float32)
        ge = tl.load(ge_ptr + b * s_eb + te[:, None] * s_et + h * D + d[None, :], mask=me[:, None], other=0.0).to(tl.float32)
        g = tl.where(is_v[:, None], gv, ge)
        r1 = tl.load(d1_ptr + b * s_db + t, mask=in_l, other=-1)
        r2 = tl.load(d2_ptr + b * s_db + t, mask=in_l, other=-1)
        m1 = in_l & (r1 >= 0)
        m2 = in_l & (r2 >= 0)
        off1 = tl.where(m1, r1, 0)[:, None] * s_g1t + h * s_g1h + d[None, :]
        off2 = tl.where(m2, r2, 0)[:, None] * s_g2t + h * s_g2h + d[None, :]
        if OUT_BF16:
            tl.store(g1_ptr.to(tl.pointer_type(tl.int16)) + off1, _rtne_bf16_bits(g), mask=m1[:, None])
            tl.store(g2_ptr.to(tl.pointer_type(tl.int16)) + off2, _rtne_bf16_bits(g), mask=m2[:, None])
        else:
            tl.store(g1_ptr + off1, g, mask=m1[:, None])
            tl.store(g2_ptr + off2, g, mask=m2[:, None])


def _use_triton(t: torch.Tensor) -> bool:
    return triton is not None and t.is_cuda


# --------------------------------------------------------------------------------------------------------------
# torch implementation of exactly the kernel math (CPU path / reference for the Triton path)
# --------------------------------------------------------------------------------------------------------------
def _rope_fwd(x, cos, sin):
    half = x.shape[-1] // 2
    c = cos[..., :half].to(x.dtype).unsqueeze(2)
    s = sin[..., :half].to(x.dtype).unsqueeze(2)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def _rope_bwd(g, cos, sin):
    half = g.shape[-1] // 2
    c = cos[..., :half].to(g.dtype).unsqueeze(2)
    s = sin[..., :half].to(g.dtype).unsqueeze(2)
    g1, g2 = g[..., :half], g[..., half:]
    return torch.cat([g1 * c + g2 * s, g2 * c - g1 * s], dim=-1)


def _cdt(x: torch.Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def _pack_fwd_torch(xv, xe, cos, sin, d1, d2, rows1, rows2, rope, out_dtype):
    cdt = _cdt(xe)
    x = torch.cat([xv.to(cdt), xe.to(cdt)], dim=1)                     # [B, L, H, D]
    if rope:
        x = _rope_fwd(x, cos, sin)
    B, L, H, D = x.shape
    outs = []
    for dst, rows in ((d1, rows1), (d2, rows2)):
        out = torch.zeros(rows, H, D, dtype=out_dtype, device=x.device)
        m = dst >= 0
        out[dst[m].long()] = x[m].to(out_dtype)
        outs.append(out)
    return outs[0], outs[1]


def _pack_bwd_torch(g1, g2, cos, sin, d1, d2, Lv, L, H, D, rope, dv_dtype, de_dtype):
    cdt = _cdt(g1)
    B = d1.shape[0]
    g = torch.zeros(B, L, H, D, dtype=cdt, device=g1.device)
    for gg, dst in ((g1, d1), (g2, d2)):
        m = dst >= 0
        g[m] += gg[dst[m].long()].to(cdt)
    if rope:
        g = _rope_bwd(g, cos, sin)
    return g[:, :Lv].to(dv_dtype).contiguous(), g[:, Lv:].to(de_dtype).contiguous()


def _unpack_fwd_torch(o1, o2, d1, d2, Lv, Le, vlm_dtype, exp_dtype):
    cdt = _cdt(o1)
    B, L = d1.shape
    H, D = o1.shape[1], o1.shape[2]
    x = torch.zeros(B, L, H, D, dtype=cdt, device=o1.device)
    for oo, dst in ((o1, d1), (o2, d2)):
        m = dst >= 0
        x[m] = oo[dst[m].long()].to(cdt)
    att = x.reshape(B, L, H * D)
    return att[:, :Lv].to(vlm_dtype).contiguous(), att[:, Lv:].to(exp_dtype).contiguous()


def _unpack_bwd_torch(gv, ge, d1, d2, rows1, rows2, H, D, dtype):
    cdt = _cdt(ge)
    B, L = d1.shape
    g = torch.cat([gv.to(cdt), ge.to(cdt)], dim=1).reshape(B, L, H, D)
    outs = []
    for dst, rows in ((d1, rows1), (d2, rows2)):
        out = torch.zeros(rows, H, D, dtype=dtype, device=gv.device)
        m = dst >= 0
        out[dst[m].long()] = g[m].to(dtype)
        outs.append(out)
    return outs[0], outs[1]


# --------------------------------------------------------------------------------------------------------------
# Triton launchers
# --------------------------------------------------------------------------------------------------------------
def _pack_fwd_triton(xv, xe, cos, sin, d1, d2, rows1, rows2, rope, out_dtype):
    B, Lv, H, D = xv.shape
    L = Lv + xe.shape[1]
    o1 = torch.empty(rows1, H, D, dtype=out_dtype, device=xv.device)
    o2 = torch.empty(rows2, H, D, dtype=out_dtype, device=xv.device)
    c = cos if rope else o1
    s = sin if rope else o1
    grid = lambda meta: (B * H, triton.cdiv(L, meta["BLOCK_T"]))  # noqa: E731
    _pack_fwd[grid](
        xv, xe, c, s, d1, d2, o1, o2, Lv, L, H,
        xv.stride(0), xv.stride(1), xv.stride(2),
        xe.stride(0), xe.stride(1), xe.stride(2),
        c.stride(0) if rope else 0, c.stride(1) if rope else 0,
        d1.stride(0),
        o1.stride(0), o1.stride(1), o2.stride(0), o2.stride(1),
        ROPE=rope, HALF=D // 2, BLOCK_T=_BLOCK_T, OUT_BF16=(out_dtype == torch.bfloat16), num_warps=_NUM_WARPS,
        enable_fp_fusion=False,  # no FMA contraction (see module docstring)
    )
    return o1, o2


def _pack_bwd_triton(g1, g2, cos, sin, d1, d2, Lv, L, H, D, rope, dv_dtype, de_dtype):
    if g1.stride(2) != 1:
        g1 = g1.contiguous()
    if g2.stride(2) != 1:
        g2 = g2.contiguous()
    B = d1.shape[0]
    Le = L - Lv
    dv = torch.empty(B, Lv, H, D, dtype=dv_dtype, device=g1.device)
    de = torch.empty(B, Le, H, D, dtype=de_dtype, device=g1.device)
    c = cos if rope else g1
    s = sin if rope else g1
    grid = lambda meta: (B * H, triton.cdiv(L, meta["BLOCK_T"]))  # noqa: E731
    _pack_bwd[grid](
        g1, g2, c, s, d1, d2, dv, de, Lv, L, H,
        g1.stride(0), g1.stride(1), g2.stride(0), g2.stride(1),
        c.stride(0) if rope else 0, c.stride(1) if rope else 0,
        d1.stride(0),
        dv.stride(0), dv.stride(1), dv.stride(2),
        de.stride(0), de.stride(1), de.stride(2),
        ROPE=rope, HALF=D // 2, BLOCK_T=_BLOCK_T, VLM_BF16=(dv_dtype == torch.bfloat16), num_warps=_NUM_WARPS,
        enable_fp_fusion=False,  # no FMA contraction (see module docstring)
    )
    return dv, de


def _unpack_fwd_triton(o1, o2, d1, d2, Lv, Le, vlm_dtype, exp_dtype):
    if o1.stride(2) != 1:
        o1 = o1.contiguous()
    if o2.stride(2) != 1:
        o2 = o2.contiguous()
    B, L = d1.shape
    H, D = o1.shape[1], o1.shape[2]
    v = torch.empty(B, Lv, H * D, dtype=vlm_dtype, device=o1.device)
    e = torch.empty(B, Le, H * D, dtype=exp_dtype, device=o1.device)
    grid = lambda meta: (B * H, triton.cdiv(L, meta["BLOCK_T"]))  # noqa: E731
    _unpack_fwd[grid](
        o1, o2, d1, d2, v, e, Lv, L, H,
        o1.stride(0), o1.stride(1), o2.stride(0), o2.stride(1),
        d1.stride(0),
        v.stride(0), v.stride(1), e.stride(0), e.stride(1),
        D=D, BLOCK_T=_BLOCK_T, VLM_BF16=(vlm_dtype == torch.bfloat16), EXP_BF16=(exp_dtype == torch.bfloat16),
        num_warps=_NUM_WARPS,
        enable_fp_fusion=False,  # no FMA contraction (see module docstring)
    )
    return v, e


def _unpack_bwd_triton(gv, ge, d1, d2, rows1, rows2, H, D, dtype):
    gv = gv.contiguous()
    ge = ge.contiguous()
    B, L = d1.shape
    Lv = gv.shape[1]
    g1 = torch.empty(rows1, H, D, dtype=dtype, device=gv.device)
    g2 = torch.empty(rows2, H, D, dtype=dtype, device=gv.device)
    grid = lambda meta: (B * H, triton.cdiv(L, meta["BLOCK_T"]))  # noqa: E731
    _unpack_bwd[grid](
        gv, ge, d1, d2, g1, g2, Lv, L, H,
        gv.stride(0), gv.stride(1), ge.stride(0), ge.stride(1),
        d1.stride(0),
        g1.stride(0), g1.stride(1), g2.stride(0), g2.stride(1),
        D=D, BLOCK_T=_BLOCK_T, OUT_BF16=(dtype == torch.bfloat16), num_warps=_NUM_WARPS,
        enable_fp_fusion=False,  # no FMA contraction (see module docstring)
    )
    return g1, g2


# --------------------------------------------------------------------------------------------------------------
# autograd Functions
# --------------------------------------------------------------------------------------------------------------
def _check_pack_inputs(xv, xe, cos, sin, d1, d2):
    assert xv.dim() == 4 and xe.dim() == 4, (xv.shape, xe.shape)
    B, Lv, H, D = xv.shape
    assert xe.shape[0] == B and xe.shape[2] == H and xe.shape[3] == D, (xv.shape, xe.shape)
    assert D % 2 == 0 and xv.stride(3) == 1 and xe.stride(3) == 1, "last dim must be contiguous"
    L = Lv + xe.shape[1]
    assert d1.shape == (B, L) and d2.shape == (B, L) and d1.dtype == torch.int32 and d2.dtype == torch.int32
    assert d1.is_contiguous() and d2.is_contiguous()
    if cos is not None:
        assert cos.shape == (B, L, D) and sin.shape == (B, L, D), (cos.shape, sin.shape, (B, L, D))
        assert cos.stride(2) == 1 and sin.stride(2) == 1


class _Pack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xv, xe, cos, sin, d1, d2, rows1, rows2, rope, out_dtype):
        _check_pack_inputs(xv, xe, cos if rope else None, sin if rope else None, d1, d2)
        ctx.save_for_backward(cos, sin, d1, d2) if rope else ctx.save_for_backward(d1, d2)
        ctx.rope = rope
        ctx.Lv, ctx.L, ctx.H, ctx.D = xv.shape[1], xv.shape[1] + xe.shape[1], xv.shape[2], xv.shape[3]
        ctx.dtypes = (xv.dtype, xe.dtype)
        f = _pack_fwd_triton if _use_triton(xv) else _pack_fwd_torch
        return f(xv, xe, cos, sin, d1, d2, rows1, rows2, rope, out_dtype)

    @staticmethod
    def backward(ctx, g1, g2):
        if ctx.rope:
            cos, sin, d1, d2 = ctx.saved_tensors
        else:
            d1, d2 = ctx.saved_tensors
            cos = sin = None
        dv_dtype, de_dtype = ctx.dtypes
        f = _pack_bwd_triton if _use_triton(g1) else _pack_bwd_torch
        dv, de = f(g1, g2, cos, sin, d1, d2, ctx.Lv, ctx.L, ctx.H, ctx.D, ctx.rope, dv_dtype, de_dtype)
        return dv, de, None, None, None, None, None, None, None, None


class _Unpack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, o1, o2, d1, d2, Lv, Le, vlm_dtype, exp_dtype):
        assert o1.dim() == 3 and o2.dim() == 3 and o1.shape[1:] == o2.shape[1:], (o1.shape, o2.shape)
        assert o1.dtype == o2.dtype
        ctx.save_for_backward(d1, d2)
        ctx.rows = (o1.shape[0], o2.shape[0])
        ctx.H, ctx.D, ctx.dtype = o1.shape[1], o1.shape[2], o1.dtype
        f = _unpack_fwd_triton if _use_triton(o1) else _unpack_fwd_torch
        return f(o1, o2, d1, d2, Lv, Le, vlm_dtype, exp_dtype)

    @staticmethod
    def backward(ctx, gv, ge):
        d1, d2 = ctx.saved_tensors
        B, L = d1.shape
        if gv is None:
            gv = torch.zeros(B, L - ge.shape[1], ctx.H * ctx.D, dtype=ctx.dtype, device=ge.device)
        if ge is None:
            ge = torch.zeros(B, L - gv.shape[1], ctx.H * ctx.D, dtype=ctx.dtype, device=gv.device)
        f = _unpack_bwd_triton if _use_triton(gv) else _unpack_bwd_torch
        g1, g2 = f(gv, ge, d1, d2, ctx.rows[0], ctx.rows[1], ctx.H, ctx.D, ctx.dtype)
        return g1, g2, None, None, None, None, None, None


# --------------------------------------------------------------------------------------------------------------
# attention: FlashAttention-2 varlen on CUDA, equivalent torch varlen attention on CPU / when forced
# --------------------------------------------------------------------------------------------------------------
def _varlen_attn_torch(q, k, v, cu_q, cu_k, causal: bool, scale: float):
    """q [Tq, Hq, D], k/v [Tk, Hkv, D]; per sample b rows cu_q[b]:cu_q[b+1] attend to cu_k[b]:cu_k[b+1].
    GQA: q head h uses kv head h // (Hq // Hkv).  causal = bottom-right aligned (FA2 convention)."""
    Hq, Hkv = q.shape[1], k.shape[1]
    rep = Hq // Hkv
    out = torch.zeros_like(q)
    cu_q_l, cu_k_l = cu_q.tolist(), cu_k.tolist()
    for b in range(len(cu_q_l) - 1):
        q0, q1, k0, k1 = cu_q_l[b], cu_q_l[b + 1], cu_k_l[b], cu_k_l[b + 1]
        lq, lk = q1 - q0, k1 - k0
        if lq == 0 or lk == 0:
            continue
        qb = q[q0:q1].transpose(0, 1)                                   # [Hq, lq, D]
        kb = k[k0:k1].transpose(0, 1).repeat_interleave(rep, dim=0)     # [Hq, lk, D]
        vb = v[k0:k1].transpose(0, 1).repeat_interleave(rep, dim=0)
        s = torch.matmul(qb, kb.transpose(-1, -2)) * scale              # [Hq, lq, lk]
        if causal:
            iq = torch.arange(lq, device=q.device)[:, None]
            ik = torch.arange(lk, device=q.device)[None, :]
            s = s.masked_fill(ik > iq + (lk - lq), float("-inf"))
        sdt = torch.float64 if s.dtype == torch.float64 else torch.float32     # softmax in fp32 like FA2 (fp64 tests stay fp64)
        p = torch.softmax(s, dim=-1, dtype=sdt).to(qb.dtype)
        out[q0:q1] = torch.matmul(p, vb).transpose(0, 1)
    return out


def _attn(q, k, v, cu_q, cu_k, max_q: int, max_k: int, causal: bool, scale: float):
    if q.is_cuda and not _FORCE_TORCH_ATTN:
        if _fa2_varlen_func is None:
            # the torch loop below is the CPU test path: host syncs, a Python loop per sample, bf16 scores
            raise RuntimeError(f"joint_fa2_attention needs flash_attn on CUDA (import failed: {_FA2_IMPORT_ERROR!r})")
        return _fa2_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k, dropout_p=0.0, softmax_scale=scale, causal=causal)
    if q.is_cuda and not getattr(_attn, "_warned", False):
        _attn._warned = True
        warnings.warn("JOINT_FA2_FORCE_TORCH=1: joint_fa2_attention runs the slow torch reference on CUDA")
    return _varlen_attn_torch(q, k, v, cu_q, cu_k, causal, scale)


@torch.compiler.disable
def joint_fa2_attention(
    q_vlm: torch.Tensor, k_vlm: torch.Tensor, v_vlm: torch.Tensor,
    q_exp: torch.Tensor, k_exp: torch.Tensor, v_exp: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    plan: FA2Plan,
    attn_dtype: torch.dtype = torch.bfloat16,
    vlm_dtype: torch.dtype = torch.bfloat16,
    exp_dtype: torch.dtype = torch.float32,
    scaling: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """q/k/v_vlm [B, Lv, Hq|Hkv, D] (bf16, after q/k norm), q/k/v_exp [B, Le, Hq|Hkv, D] (fp32),
    cos/sin [B, L, D] fp32.  Returns att_vlm [B, Lv, Hq*D] (vlm_dtype, pad rows 0), att_exp [B, Le, Hq*D] (exp_dtype)."""
    D = q_vlm.shape[3]
    scale = D ** -0.5 if scaling is None else scaling
    qA, qB = _Pack.apply(q_vlm, q_exp, cos, sin, plan.dstA, plan.dstBq, plan.rowsA, plan.rowsBq, True, attn_dtype)
    kA, kB = _Pack.apply(k_vlm, k_exp, cos, sin, plan.dstA, plan.dstBk, plan.rowsA, plan.rowsBk, True, attn_dtype)
    vA, vB = _Pack.apply(v_vlm, v_exp, None, None, plan.dstA, plan.dstBk, plan.rowsA, plan.rowsBk, False, attn_dtype)
    oA = _attn(qA, kA, vA, plan.cuA, plan.cuA, plan.max_a, plan.max_a, True, scale)
    oB = _attn(qB, kB, vB, plan.cuBq, plan.cuBk, plan.max_bq, plan.max_bk, False, scale)
    return _Unpack.apply(oA, oB, plan.dstA, plan.dstBq, plan.Lv, plan.Le, vlm_dtype, exp_dtype)


# --------------------------------------------------------------------------------------------------------------
# Reference: dense masked attention with the current eager input chain (joint_qkv_prep_reference) and output chain
# --------------------------------------------------------------------------------------------------------------
def joint_attention_reference(
    q_vlm, k_vlm, v_vlm, q_exp, k_exp, v_exp, cos, sin, att_2d_masks,
    vlm_dtype: torch.dtype = torch.bfloat16, exp_dtype: torch.dtype = torch.float32,
    scaling: Optional[float] = None,
):
    """Dense attention over all L tokens in fp32 (fp64 if the expert inputs are fp64) with the model's mask.
    Fully masked (pad) query rows give 0.  Output split like joint_attn_split: att_vlm (vlm_dtype), att_exp (exp_dtype)."""
    from lingbotvla.ops.joint_qkv_prep import joint_qkv_prep_reference

    Q, K, V = joint_qkv_prep_reference(q_vlm, k_vlm, v_vlm, q_exp, k_exp, v_exp, cos, sin)   # [B, H, L, D]
    B, Hq, L, D = Q.shape
    Hkv = K.shape[1]
    rep = Hq // Hkv
    K = K.repeat_interleave(rep, dim=1)
    V = V.repeat_interleave(rep, dim=1)
    scale = D ** -0.5 if scaling is None else scaling
    mask = att_2d_masks.bool()
    row_valid = mask.any(-1)                                                                   # [B, L]
    eye = torch.eye(L, dtype=torch.bool, device=mask.device)[None]
    mask_safe = mask | (~row_valid)[:, :, None] & eye                                          # keep softmax finite
    S = torch.matmul(Q, K.transpose(-1, -2)) * scale
    S = S.masked_fill(~mask_safe[:, None], float("-inf"))
    P = torch.softmax(S, dim=-1)
    O = torch.matmul(P, V) * row_valid[:, None, :, None].to(Q.dtype)                           # pad rows -> 0
    att = O.transpose(1, 2).reshape(B, L, Hq * D)
    Lv = q_vlm.shape[1]
    return att[:, :Lv].to(vlm_dtype).contiguous(), att[:, Lv:].to(exp_dtype).contiguous()
