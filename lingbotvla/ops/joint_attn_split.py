"""Fused split of the joint flex-attention output back into the VLM and expert streams.

On the unfused path (flex_attention_with_block_mask's output slice / transpose / reshape, then the output_atten
branch of Qwen3VLTextDecoderLayer.forward and Qwen2DecoderLayer.forward: .to(param_dtype), att_output[:, start:end])
the attention output O [B, Hq, L_pad, D] fp32 goes through:
    O[:, :, :L].transpose(1, 2).contiguous().reshape(B, L, Hq*D)   # 1 copy (fp32, all tokens)
    att.to(bf16)                                                     # VLM layer: cast of ALL 337 tokens
    att[:, 0:Lv]  / att[:, Lv:L]                                     # per-stream slice -> contiguous copies
and in backward each slice produces a full-size zero tensor plus a copy, the two are added, the cast is undone.
Per layer that is ~0.75 ms forward, ~0.4 ms in recompute and ~2.4 ms backward of pure data movement.

`joint_attn_split(O, Lv, Le)` does it in one kernel per direction:
    att_vlm [B, Lv, Hq*D]  (dtype = vlm_dtype, default bf16, RTNE)   att_exp [B, Le, Hq*D]  (O.dtype)
    backward: dO [B, Hq, L_pad, D] in O.dtype, rows >= Lv+Le (padding, if any) zero-filled.
Both directions are bit-identical to the eager chain (pure copies + one RTNE rounding).  The two layer calls then
receive their own tensor with start=0, end=L_i, so the in-layer `.to(param_dtype)` and `[:, start:end]` become no-ops.

On CUDA the Triton kernels run; on CPU (tests) an equivalent torch implementation runs.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

_BLOCK_T = int(os.environ.get("JOINT_ATTN_SPLIT_BLOCK_T", "16"))
_NUM_WARPS = int(os.environ.get("JOINT_ATTN_SPLIT_NUM_WARPS", "8"))


if triton is not None:

    @triton.jit
    def _rtne_bf16_bits(x):
        u = x.to(tl.int32, bitcast=True)
        lsb = (u >> 16) & 1
        r = (u + 0x7FFF + lsb) >> 16
        # NaN stays NaN (0x7FFF, which CUDA's cvt.rn.bf16.f32, i.e. eager .to(bfloat16) on the GPU, gives); the carry
        # above would turn it into +-0 or +-inf. Integer test: Triton compiles float compares assuming no NaNs.
        r = tl.where((u & 0x7FFFFFFF) > 0x7F800000, 0x7FFF, r)
        return r.to(tl.int16)

    @triton.jit
    def _split_fwd(o_ptr, v_ptr, e_ptr, Lv, L, H,
                   s_ob, s_oh, s_ot,          # O [B, H, L_pad, D] strides (d stride 1)
                   s_vb, s_vt, s_eb, s_et,    # outputs [B, L_i, H*D]: token stride, head offset = h*D
                   VLM_BF16: tl.constexpr, EXP_BF16: tl.constexpr, D: tl.constexpr, BLOCK_T: tl.constexpr):
        pid_bh = tl.program_id(0)
        b = pid_bh // H
        h = pid_bh - b * H
        t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
        d = tl.arange(0, D)
        in_l = t < L
        is_v = t < Lv
        mv = in_l & is_v
        me = in_l & (t >= Lv)
        x = tl.load(o_ptr + b * s_ob + h * s_oh + t[:, None] * s_ot + d[None, :], mask=in_l[:, None], other=0.0).to(tl.float32)
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
    def _split_bwd(gv_ptr, ge_ptr, go_ptr, Lv, L, Lpad, H,
                   s_vb, s_vt, s_eb, s_et,
                   s_ob, s_oh, s_ot,
                   OUT_BF16: tl.constexpr, D: tl.constexpr, BLOCK_T: tl.constexpr):
        pid_bh = tl.program_id(0)
        b = pid_bh // H
        h = pid_bh - b * H
        t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
        d = tl.arange(0, D)
        in_pad = t < Lpad
        is_v = t < Lv
        in_e = (t >= Lv) & (t < L)
        tv = tl.where(is_v, t, 0)
        te = tl.where(in_e, t - Lv, 0)
        gv = tl.load(gv_ptr + b * s_vb + tv[:, None] * s_vt + h * D + d[None, :], mask=is_v[:, None], other=0.0).to(tl.float32)
        ge = tl.load(ge_ptr + b * s_eb + te[:, None] * s_et + h * D + d[None, :], mask=in_e[:, None], other=0.0).to(tl.float32)
        g = tl.where(is_v[:, None], gv, ge)       # rows >= L stay 0 (padding)
        offo = b * s_ob + h * s_oh + t[:, None] * s_ot + d[None, :]
        if OUT_BF16:
            tl.store(go_ptr.to(tl.pointer_type(tl.int16)) + offo, _rtne_bf16_bits(g), mask=in_pad[:, None])
        else:
            tl.store(go_ptr + offo, g, mask=in_pad[:, None])


def _fwd_torch(o, Lv, Le, vlm_dtype, exp_dtype):
    B, H, Lp, D = o.shape
    att = o[:, :, :Lv + Le].transpose(1, 2).reshape(B, Lv + Le, H * D)
    return att[:, :Lv].to(vlm_dtype).contiguous(), att[:, Lv:].to(exp_dtype).contiguous()


def _bwd_torch(gv, ge, Lv, Le, Lp, H, D, dtype):
    B = gv.shape[0]
    g = torch.zeros(B, Lp, H, D, dtype=dtype, device=gv.device)
    g[:, :Lv] = gv.to(dtype).view(B, Lv, H, D)
    g[:, Lv:Lv + Le] = ge.to(dtype).view(B, Le, H, D)
    return g.transpose(1, 2).contiguous()


def _fwd_triton(o, Lv, Le, vlm_dtype, exp_dtype):
    B, H, Lp, D = o.shape
    L = Lv + Le
    v = torch.empty(B, Lv, H * D, dtype=vlm_dtype, device=o.device)
    e = torch.empty(B, Le, H * D, dtype=exp_dtype, device=o.device)
    grid = lambda meta: (B * H, triton.cdiv(L, meta["BLOCK_T"]))  # noqa: E731
    _split_fwd[grid](o, v, e, Lv, L, H, o.stride(0), o.stride(1), o.stride(2), v.stride(0), v.stride(1), e.stride(0), e.stride(1),
                     VLM_BF16=(vlm_dtype == torch.bfloat16), EXP_BF16=(exp_dtype == torch.bfloat16), D=D, BLOCK_T=_BLOCK_T,
                     num_warps=_NUM_WARPS)
    return v, e


def _bwd_triton(gv, ge, Lv, Le, Lp, H, D, dtype):
    B = gv.shape[0]
    L = Lv + Le
    go = torch.empty(B, H, Lp, D, dtype=dtype, device=gv.device)
    grid = lambda meta: (B * H, triton.cdiv(Lp, meta["BLOCK_T"]))  # noqa: E731
    _split_bwd[grid](gv, ge, go, Lv, L, Lp, H, gv.stride(0), gv.stride(1), ge.stride(0), ge.stride(1),
                     go.stride(0), go.stride(1), go.stride(2), OUT_BF16=(dtype == torch.bfloat16), D=D, BLOCK_T=_BLOCK_T,
                     num_warps=_NUM_WARPS)
    return go


def _use_triton(t: torch.Tensor) -> bool:
    return triton is not None and t.is_cuda


class _JointAttnSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, o, Lv, Le, vlm_dtype, exp_dtype):
        assert o.dim() == 4 and o.stride(3) == 1 and Lv + Le <= o.shape[2], (o.shape, Lv, Le)
        ctx.Lv, ctx.Le, ctx.shape, ctx.dtype = Lv, Le, tuple(o.shape), o.dtype
        return (_fwd_triton if _use_triton(o) else _fwd_torch)(o, Lv, Le, vlm_dtype, exp_dtype)

    @staticmethod
    def backward(ctx, gv, ge):
        B, H, Lp, D = ctx.shape
        if gv is None: gv = torch.zeros(B, ctx.Lv, H * D, dtype=ctx.dtype, device=ge.device)
        if ge is None: ge = torch.zeros(B, ctx.Le, H * D, dtype=ctx.dtype, device=gv.device)
        gv = gv.contiguous(); ge = ge.contiguous()
        go = (_bwd_triton if _use_triton(gv) else _bwd_torch)(gv, ge, ctx.Lv, ctx.Le, Lp, H, D, ctx.dtype)
        return go, None, None, None, None


@torch.compiler.disable
def joint_attn_split(o: torch.Tensor, Lv: int, Le: int, vlm_dtype: torch.dtype = torch.bfloat16,
                     exp_dtype: Optional[torch.dtype] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """o: flex output [B, Hq, L_pad, D] (L_pad >= Lv+Le). Returns att_vlm [B, Lv, Hq*D] in vlm_dtype,
    att_exp [B, Le, Hq*D] in exp_dtype (default: o.dtype; with a bf16 attention output pass the expert's fp32)."""
    return _JointAttnSplit.apply(o, Lv, Le, vlm_dtype, o.dtype if exp_dtype is None else exp_dtype)


def joint_attn_split_reference(o: torch.Tensor, Lv: int, Le: int, vlm_dtype: torch.dtype = torch.bfloat16,
                               exp_dtype: Optional[torch.dtype] = None):
    """The unfused eager chain (flex_attention_with_block_mask's output slice + the two layers' .to(param_dtype) and
    [:, start:end])."""
    B, H, Lp, D = o.shape
    L = Lv + Le
    att = o[:, :, :L, :].to(o.dtype).transpose(1, 2).contiguous().reshape(B, L, H * D)
    att_v = att.to(vlm_dtype)[:, 0:Lv]
    att_e = att.to(o.dtype if exp_dtype is None else exp_dtype)[:, Lv:L]
    return att_v, att_e
