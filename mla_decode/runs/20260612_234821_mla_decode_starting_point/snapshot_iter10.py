# EVOLVE-BLOCK-START
"""
Initial MLA Decode submission — optimised baseline with Triton softmax and RoPE kernels.
"""

import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config


@triton.jit
def rope_swap_halves_kernel(
    x_ptr,
    cos_ptr, sin_ptr,
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    stride_xb, stride_xt, stride_xd,
    stride_cos_t, stride_cos_d,
    stride_sin_t, stride_sin_d,
    BLOCK_HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    bt = pid
    b = bt // T
    t = bt - b * T

    half = D // 2

    off = tl.arange(0, BLOCK_HALF)
    mask = off < half

    x_base = x_ptr + b * stride_xb + t * stride_xt
    x0_ptr = x_base + off * stride_xd
    x1_ptr = x_base + (half + off) * stride_xd

    cos_base = cos_ptr + t * stride_cos_t
    sin_base = sin_ptr + t * stride_sin_t

    c_ptr = cos_base + off * stride_cos_d
    s_ptr = sin_base + off * stride_sin_d

    x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(c_ptr, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_ptr, mask=mask, other=0.0).to(tl.float32)

    out0 = x0 * c - x1 * s
    out1 = x1 * c + x0 * s

    tl.store(x0_ptr, out0.to(tl.bfloat16), mask=mask)
    tl.store(x1_ptr, out1.to(tl.bfloat16), mask=mask)


def rope_inplace_query(q_rope: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor):
    assert q_rope.is_cuda
    assert q_rope.shape[-1] % 2 == 0
    bs, nh, d_rope = q_rope.shape

    half = d_rope // 2
    BLOCK_HALF = 1 << (half - 1).bit_length()

    grid = (bs * nh,)

    rope_swap_halves_kernel[grid](
        q_rope,
        cos_q, sin_q,
        B=bs, T=nh, D=d_rope,
        stride_xb=q_rope.stride(0),
        stride_xt=q_rope.stride(1),
        stride_xd=q_rope.stride(2),
        stride_cos_t=0, stride_cos_d=cos_q.stride(0),
        stride_sin_t=0, stride_sin_d=sin_q.stride(0),
        BLOCK_HALF=BLOCK_HALF,
        num_warps=4,
    )


_rope_cache = {}


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _get_rope_tables(dim: int, max_seq_len: int, device: torch.device):
    key = (dim, max_seq_len, device)
    if key not in _rope_cache:
        half = dim // 2
        theta = (10000.0 ** (-torch.arange(half, dtype=torch.float32, device=device) / half)).to(
            torch.bfloat16
        )
        pos = torch.arange(max_seq_len, dtype=torch.int64, device=device).unsqueeze_(1)
        idx = pos * theta[None, :]
        idx = torch.cat([idx, idx], dim=-1)
        _rope_cache[key] = (idx.cos().to(torch.bfloat16), idx.sin().to(torch.bfloat16))
    return _rope_cache[key]


@triton.jit
def _softmax_kernel(
    out_ptr, in_ptr,
    stride_out, stride_in,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_in = row * stride_in
    row_off_out = row * stride_out

    max_val = tl.full([BLOCK_SIZE], -float("inf"), tl.float32)
    col = tl.arange(0, BLOCK_SIZE)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.cast(val, tl.float32))
    row_max = tl.max(max_val)

    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(in_ptr + row_off_in + cur, mask=mask, other=-float('inf'))
        exp_val = tl.exp(tl.cast(val, tl.float32) - row_max)
        tl.store(out_ptr + row_off_out + cur, tl.cast(exp_val, tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0)
        norm = tl.cast(val, tl.float32) / row_sum
        tl.store(out_ptr + row_off_out + cur, tl.cast(norm, tl.bfloat16), mask=mask)


def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.bfloat16
    n_rows, n_cols = x.shape

    if n_cols <= 32:
        BLOCK_SIZE = 32
    elif n_cols <= 64:
        BLOCK_SIZE = 64
    elif n_cols <= 128:
        BLOCK_SIZE = 128
    else:
        BLOCK_SIZE = 1 << (n_cols - 1).bit_length()
        BLOCK_SIZE = min(BLOCK_SIZE, 1024)

    out = torch.empty_like(x)
    grid = (n_rows,)
    _softmax_kernel[grid](
        out, x,
        out.stride(0), x.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_STAGES=2,
        num_warps=4,
    )
    return out


def _attention_inner(
    q_nope_latent, q_rope, kv_nope_input, k_rope_input,
    cos_q, sin_q, cos_k, sin_k,
    wV_T, wO,
    bs, nh, dkv, d_nope, d_rope, dv, scale,
):
    """
    Compiled attention inner loop.
    q_nope_latent is already pre-projected (wK absorbed in fused weight).
    """
    # Apply RoPE to queries
    q_rope = q_rope * cos_q + torch.cat((-q_rope[..., d_rope//2:], q_rope[..., :d_rope//2]), dim=-1) * sin_q

    # Apply RoPE to keys
    k_rope_half = torch.cat((-k_rope_input[..., d_rope//2:], k_rope_input[..., :d_rope//2]), dim=-1)
    k_rope = k_rope_input * cos_k + k_rope_half * sin_k  # [bs, kv_len, d_rope]

    # Score computation (q_nope_latent already computed via fused GEMM)
    kv_nope_T = kv_nope_input.transpose(1, 2)
    scores_nope = torch.matmul(q_nope_latent, kv_nope_T)          # [bs, nh, kv_len]
    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))   # [bs, nh, kv_len]
    scores = (scores_nope + scores_rope) * scale

    # Softmax + attention output
    attn = torch.softmax(scores, dim=-1)
    M = torch.matmul(attn, kv_nope_input)  # [bs, nh, dkv]

    # Project through wV
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)  # [bs, nh, dv]
    y = y_head.reshape(bs, nh * dv).unsqueeze(1)
    output = F.linear(y, wO)
    return output


_compiled_attention = torch.compile(_attention_inner, mode='reduce-overhead', dynamic=True)

# Cache for fused weights: keyed by (wDQ_ptr, wUQ_ptr)
_fused_weight_cache = {}


def _get_fused_weights(wDQ, wUQ, wUKV, nh, dkv, d_nope, d_rope, dv):
    """
    Precompute and cache fused projection weights:
      wQ_fused = wUQ @ wDQ  shape [nh*(d_nope+d_rope), dim]
        => q_up = F.linear(x.squeeze(1), wQ_fused)  (single GEMM, no intermediate)
      wQnl_fused: per-head nope-latent absorption also fused
        For q_nope_latent = einsum('bhd,hdk->bhk', q_nope, wK):
          flatten: q_nope [bs, nh*d_nope] @ wK_flat.T where wK_flat [nh*dkv, nh*d_nope] block-diag
          Precompute wQnl = wK_flat @ wUQ_nope @ wDQ  [nh*dkv, dim]
          => q_nope_latent = F.linear(x.squeeze(1), wQnl).view(bs, nh, dkv)
    """
    key = (wDQ.data_ptr(), wUQ.data_ptr(), wUKV.data_ptr())
    if key not in _fused_weight_cache:
        wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
        wK = wUKV_view[:, :d_nope, :]  # [nh, d_nope, dkv]

        # wUQ layout: [nh*(d_nope+d_rope), dq], rows are [h0_nope, h0_rope, h1_nope, h1_rope, ...]
        # because q_up.view(bs, nh, d_nope+d_rope) splits each head's output
        wUQ_bh = wUQ.view(nh, d_nope + d_rope, -1)   # [nh, d_nope+d_rope, dq]
        wUQ_nope_bh = wUQ_bh[:, :d_nope, :]          # [nh, d_nope, dq]
        wUQ_rope_bh = wUQ_bh[:, d_nope:, :]          # [nh, d_rope, dq]

        # wQ_fused for rope part: shape [nh*d_rope, dim]
        wQ_rope_fused = wUQ_rope_bh.reshape(nh * d_rope, -1) @ wDQ  # [nh*d_rope, dim]
        # wK[h]: [d_nope, dkv], wUQ_nope_bh[h]: [d_nope, dq]
        # wK[h].T @ wUQ_nope_bh[h] = [dkv, dq]
        # Batch: [nh, dkv, dq] = wK.permute(0,2,1) @ wUQ_nope_bh
        wKt = wK.permute(0, 2, 1)              # [nh, dkv, d_nope]
        # [nh, dkv, d_nope] @ [nh, d_nope, dq] = [nh, dkv, dq]
        wKU = torch.bmm(wKt, wUQ_nope_bh)     # [nh, dkv, dq]
        # Then @ wDQ: [nh, dkv, dim]
        wKU_flat = wKU.reshape(nh * dkv, -1)  # [nh*dkv, dq]
        wQnl_fused = wKU_flat @ wDQ            # [nh*dkv, dim]

        _fused_weight_cache[key] = (
            wQ_rope_fused.contiguous(),
            wQnl_fused.contiguous(),
        )
    return _fused_weight_cache[key]


def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised MLA decode with fused projection weights:
    - wDQ+wUQ_rope fused -> single GEMM for q_rope
    - wDQ+wUQ_nope+wK fused -> single GEMM for q_nope_latent
    Eliminates intermediate q_lora and q_nope tensors, reduces GEMMs.
    """
    config, x, kv_cache = data

    bs = config.batch_size
    nh = config.n_heads
    dkv = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv = config.v_head_dim
    msl = config.max_seq_len

    wDQ = config.Q_proj_down_weight
    wDKV = config.KV_proj_down_weight
    wUQ = config.Q_proj_up_weight
    wUKV = config.KV_proj_up_weight
    wO = config.wo_weight

    # Get fused weights (precomputed + cached)
    wQ_rope_fused, wQnl_fused = _get_fused_weights(
        wDQ, wUQ, wUKV, nh, dkv, d_nope, d_rope, dv
    )

    x2d = x.squeeze(1)  # [bs, dim]

    # Fused projections: single GEMM each instead of two chained GEMMs
    q_nope_latent = F.linear(x2d, wQnl_fused).view(bs, nh, dkv)   # [bs, nh, dkv]
    q_rope_raw    = F.linear(x2d, wQ_rope_fused).view(bs, nh, d_rope)  # [bs, nh, d_rope]

    # KV projection (unchanged)
    kv_lora_input = F.linear(x2d, wDKV)  # [bs, 576]
    kv_lora, kv_len = kv_cache(kv_lora_input.unsqueeze(1))
    query_pos = kv_len - 1

    kv_nope_input = kv_lora[..., :dkv]    # [bs, kv_len, dkv]
    k_rope_input  = kv_lora[..., dkv:]    # [bs, kv_len, d_rope]

    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)
    cos_q = cos_table[query_pos]
    sin_q = sin_table[query_pos]
    cos_k = cos_table[:kv_len]
    sin_k = sin_table[:kv_len]

    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
    wV = wUKV_view[:, d_nope:, :]
    wV_T = wV.permute(0, 2, 1)  # [nh, dkv, dv]

    scale = 1.0 / math.sqrt(d_nope + d_rope)

    # Attention (compiled) — q_nope_latent already computed via fused GEMM
    output = _compiled_attention(
        q_nope_latent, q_rope_raw, kv_nope_input, k_rope_input,
        cos_q, sin_q, cos_k, sin_k,
        wV_T, wO,
        bs, nh, dkv, d_nope, d_rope, dv, scale,
    )

    return output, kv_cache.data
# EVOLVE-BLOCK-END
