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
_fused_weight_cache = {}  # cache for precomputed fused q_nope_latent weight


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


def _get_fused_weights(wUQ, wUKV, nh, d_nope, d_rope, dkv, dv):
    """
    Precompute:
      1. W_nope_fused [dq, nh*dkv]: fuses wUQ_nope + wK absorption into single GEMM
      2. W_rope [nh*d_rope, dq]: extracted rope projection weight (transposed for F.linear)
    Both cached by weight tensor identity.
    """
    key = (id(wUQ), id(wUKV))
    if key not in _fused_weight_cache:
        wUKV_view = wUKV.view(nh, d_nope + dv, dkv)      # [nh, d_nope+dv, dkv]
        wK = wUKV_view[:, :d_nope, :]                      # [nh, d_nope, dkv]
        wUQ_view = wUQ.view(nh, d_nope + d_rope, -1)       # [nh, d_nope+d_rope, dq]
        dq = wUQ_view.shape[2]

        # W_nope_fused: q_lora [bs,dq] @ W_nope_fused [dq, nh*dkv] -> q_nope_latent [bs, nh*dkv]
        wUQ_nope = wUQ_view[:, :d_nope, :]                 # [nh, d_nope, dq]
        W_fused_heads = torch.bmm(
            wUQ_nope.permute(0, 2, 1),                     # [nh, dq, d_nope]
            wK                                              # [nh, d_nope, dkv]
        )                                                   # [nh, dq, dkv]
        W_nope_fused = W_fused_heads.permute(1, 0, 2).reshape(dq, nh * dkv).contiguous()

        # W_rope: F.linear weight [nh*d_rope, dq] for extracting rope queries
        wUQ_rope = wUQ_view[:, d_nope:, :].reshape(nh * d_rope, dq).contiguous()

        _fused_weight_cache[key] = (W_nope_fused, wUQ_rope)

    return _fused_weight_cache[key]


def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised MLA decode. Fuses wUQ_nope + wK absorption into a single GEMM.
    q_lora @ W_fused -> q_nope_latent in one shot, eliminating einsum overhead.
    """
    config, x, kv_cache = data

    bs = config.batch_size
    sl = config.seq_len
    nh = config.n_heads
    dq = config.q_lora_rank
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

    q_lora = F.linear(x, wDQ)           # [bs, 1, dq]
    kv_lora_input = F.linear(x, wDKV)

    kv_lora, kv_len = kv_cache(kv_lora_input)
    query_pos = kv_len - 1

    kv_nope_input = kv_lora[..., :dkv]
    k_rope_input = kv_lora[..., dkv:]

    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # RoPE for keys
    cos_k = cos_table[:kv_len]
    sin_k = sin_table[:kv_len]
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k

    # ---- Fused query projections (cached weights) ----
    W_nope_fused, W_rope = _get_fused_weights(wUQ, wUKV, nh, d_nope, d_rope, dkv, dv)
    q_lora_2d = q_lora.squeeze(1)                                      # [bs, dq]
    # Single GEMM: q_lora_2d [bs, dq] @ W_nope_fused [dq, nh*dkv] -> [bs, nh*dkv]
    q_nope_latent = (q_lora_2d @ W_nope_fused).view(bs, nh, dkv)      # [bs, nh, dkv]
    # Rope projection: q_lora_2d [bs,dq] @ W_rope.T -> [bs, nh*d_rope]
    q_rope = F.linear(q_lora_2d, W_rope).view(bs, nh, d_rope)         # [bs, nh, d_rope]

    # RoPE for query
    cos_q = cos_table[query_pos].view(d_rope)
    sin_q = sin_table[query_pos].view(d_rope)
    q_rope = q_rope * cos_q + _rotate_half(q_rope) * sin_q

    # ---- Attention ----
    kv_nope_T = kv_nope_input.transpose(1, 2)
    scores_nope = torch.matmul(q_nope_latent, kv_nope_T)
    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))

    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale

    scores_flat = scores.reshape(bs * nh, kv_len)
    attn_flat = _triton_softmax(scores_flat)
    attn = attn_flat.view(bs, nh, kv_len)

    M = torch.matmul(attn, kv_nope_input)

    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
    wV = wUKV_view[:, d_nope:, :]
    wV_T = wV.permute(0, 2, 1)
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)

    y = y_head.reshape(bs, nh * dv).unsqueeze(1)
    output = F.linear(y, wO)

    return output, kv_cache.data
# EVOLVE-BLOCK-END
