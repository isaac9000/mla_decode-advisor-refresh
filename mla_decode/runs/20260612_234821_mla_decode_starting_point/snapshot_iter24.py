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
    q_nope, q_rope, kv_nope_input, kv_nope_T, k_rope_input, k_rope_neg,
    cos_q, sin_q, cos_k, sin_k,
    wK, wV_T, wO,
    bs, nh, dkv, d_nope, d_rope, dv, scale,
):
    """
    Compiled attention inner loop.
    Cleanups: removed unused wUQ, pre-transposed kv_nope, pre-computed k_rope_neg.
    Removes two torch.cat ops from compiled graph — one per RoPE application.
    """
    # Apply RoPE to queries (one torch.cat remains — q_rope_neg not pre-computed)
    q_rope = q_rope * cos_q + torch.cat((-q_rope[..., d_rope//2:], q_rope[..., :d_rope//2]), dim=-1) * sin_q

    # Apply RoPE to keys (k_rope_neg pre-computed outside — no torch.cat here)
    k_rope = k_rope_input * cos_k + k_rope_neg * sin_k

    # Absorb wK
    q_nope_latent = torch.einsum("bhd,hdk->bhk", q_nope, wK)

    # Score computation (kv_nope_T pre-transposed outside)
    scores_nope = torch.matmul(q_nope_latent, kv_nope_T)
    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))
    scores = (scores_nope + scores_rope) * scale

    # Softmax + attention output
    attn = torch.softmax(scores, dim=-1)
    M = torch.matmul(attn, kv_nope_input)

    # Project through wV then wO
    y_head = torch.einsum("bhd,hdk->bhk", M, wV_T)
    y = y_head.reshape(bs, nh * dv).unsqueeze(1)
    return F.linear(y, wO)


_compiled_attention = torch.compile(
    _attention_inner, mode="max-autotune-no-cudagraphs", dynamic=True
)


def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised MLA decode: exp #12 structure with two cleanups:
    - removed unused wUQ argument from compiled function
    - pre-transpose kv_nope outside compiled scope (free metadata op)
    """
    config, x, kv_cache = data

    bs = config.batch_size
    nh = config.n_heads
    dkv = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv = config.v_head_dim
    msl = config.max_seq_len

    wDQ  = config.Q_proj_down_weight
    wDKV = config.KV_proj_down_weight
    wUQ  = config.Q_proj_up_weight
    wUKV = config.KV_proj_up_weight
    wO   = config.wo_weight

    q_lora = F.linear(x, wDQ)
    kv_lora_input = F.linear(x, wDKV)

    kv_lora, kv_len = kv_cache(kv_lora_input)
    query_pos = kv_len - 1

    q_up = F.linear(q_lora.squeeze(1), wUQ)
    q_up = q_up.view(bs, nh, d_nope + d_rope)
    q_nope = q_up[..., :d_nope]
    q_rope = q_up[..., d_nope:]

    kv_nope_input = kv_lora[..., :dkv]
    k_rope_input  = kv_lora[..., dkv:]

    # Pre-compute ops outside compiled scope (free/cheap metadata ops)
    kv_nope_T = kv_nope_input.transpose(1, 2)
    # Pre-compute negated-swapped k_rope for RoPE: avoids torch.cat inside compiled graph
    k_rope_neg = torch.cat((-k_rope_input[..., d_rope//2:], k_rope_input[..., :d_rope//2]), dim=-1)

    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)
    cos_q = cos_table[query_pos]
    sin_q = sin_table[query_pos]
    cos_k = cos_table[:kv_len]
    sin_k = sin_table[:kv_len]

    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
    wK   = wUKV_view[:, :d_nope, :]
    wV   = wUKV_view[:, d_nope:, :]
    wV_T = wV.permute(0, 2, 1)

    scale = 1.0 / math.sqrt(d_nope + d_rope)

    output = _compiled_attention(
        q_nope, q_rope, kv_nope_input, kv_nope_T, k_rope_input, k_rope_neg,
        cos_q, sin_q, cos_k, sin_k,
        wK, wV_T, wO,
        bs, nh, dkv, d_nope, d_rope, dv, scale,
    )

    return output, kv_cache.data
# EVOLVE-BLOCK-END