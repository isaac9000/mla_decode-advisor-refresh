# EVOLVE-BLOCK-START
"""
MLA Decode submission — Triton flash-decode attention in compressed latent space.
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
def flash_decode_latent_kernel(
    # Inputs
    q_nope_ptr,    # [bs, nh, DKV]
    q_rope_ptr,    # [bs, nh, DROPE]
    kv_nope_ptr,   # [bs, kv_len, DKV]
    k_rope_ptr,    # [bs, kv_len, DROPE]
    # Output
    out_ptr,       # [bs, nh, DKV]
    # Strides
    stride_qn_b, stride_qn_h, stride_qn_d,
    stride_qr_b, stride_qr_h, stride_qr_d,
    stride_kn_b, stride_kn_t, stride_kn_d,
    stride_kr_b, stride_kr_t, stride_kr_d,
    stride_o_b, stride_o_h, stride_o_d,
    # Dims
    kv_len,
    DKV: tl.constexpr,
    DROPE: tl.constexpr,
    scale,
    BLOCK_KV: tl.constexpr,
    BLOCK_DKV: tl.constexpr,
    BLOCK_DROPE: tl.constexpr,
):
    # 2D grid: axis 0 = batch, axis 1 = head
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Load q_nope: [DKV]
    q_nope_base = q_nope_ptr + pid_b * stride_qn_b + pid_h * stride_qn_h
    q_rope_base = q_rope_ptr + pid_b * stride_qr_b + pid_h * stride_qr_h

    # Load full q_nope vector into registers (DKV=512, BLOCK_DKV must cover it)
    offs_dkv = tl.arange(0, BLOCK_DKV)
    q_nope = tl.load(q_nope_base + offs_dkv * stride_qn_d, mask=offs_dkv < DKV, other=0.0).to(tl.float32)

    offs_drope = tl.arange(0, BLOCK_DROPE)
    q_rope = tl.load(q_rope_base + offs_drope * stride_qr_d, mask=offs_drope < DROPE, other=0.0).to(tl.float32)

    # Online softmax state
    m_i = tl.full([1], float('-inf'), tl.float32)
    l_i = tl.full([1], 0.0, tl.float32)
    acc = tl.zeros([BLOCK_DKV], tl.float32)

    kv_nope_base = kv_nope_ptr + pid_b * stride_kn_b
    k_rope_base = k_rope_ptr + pid_b * stride_kr_b

    for start_t in range(0, kv_len, BLOCK_KV):
        offs_t = start_t + tl.arange(0, BLOCK_KV)
        mask_t = offs_t < kv_len

        # Load k_nope tile: [BLOCK_KV, DKV]
        kn_ptrs = kv_nope_base + offs_t[:, None] * stride_kn_t + offs_dkv[None, :] * stride_kn_d
        k_nope_tile = tl.load(kn_ptrs, mask=mask_t[:, None] & (offs_dkv[None, :] < DKV), other=0.0).to(tl.float32)

        # Load k_rope tile: [BLOCK_KV, DROPE]
        kr_ptrs = k_rope_base + offs_t[:, None] * stride_kr_t + offs_drope[None, :] * stride_kr_d
        k_rope_tile = tl.load(kr_ptrs, mask=mask_t[:, None] & (offs_drope[None, :] < DROPE), other=0.0).to(tl.float32)

        # Compute scores: [BLOCK_KV]
        scores_nope = tl.sum(k_nope_tile * q_nope[None, :], axis=1)   # [BLOCK_KV]
        scores_rope = tl.sum(k_rope_tile * q_rope[None, :], axis=1)   # [BLOCK_KV]
        scores = (scores_nope + scores_rope) * scale
        scores = tl.where(mask_t, scores, float('-inf'))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores))
        exp_scores = tl.exp(scores - m_new)
        exp_old = tl.exp(m_i - m_new)
        l_new = exp_old * l_i + tl.sum(exp_scores)

        # Accumulate: acc = acc * (exp_old * l_i / l_new) + sum(exp_scores * k_nope_tile) / l_new
        # We keep acc unnormalized by l, normalize at the end
        acc = acc * (exp_old * l_i / l_new) + tl.sum(exp_scores[:, None] * k_nope_tile, axis=0) / l_new

        m_i = m_new
        l_i = l_new

    # Store output
    out_base = out_ptr + pid_b * stride_o_b + pid_h * stride_o_h
    tl.store(out_base + offs_dkv * stride_o_d, acc.to(tl.bfloat16), mask=offs_dkv < DKV)


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


def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward step of the Multi-head Latent Attention (MLA) module.
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

    q_lora = F.linear(x, wDQ)
    kv_lora_input = F.linear(x, wDKV)

    kv_lora, kv_len = kv_cache(kv_lora_input)
    query_pos = kv_len - 1

    q_up = F.linear(q_lora.squeeze(1), wUQ)
    q_up = q_up.view(bs, nh, d_nope + d_rope)
    q_nope = q_up[..., :d_nope]
    q_rope = q_up[..., d_nope:]

    kv_nope_input = kv_lora[..., :dkv]
    k_rope_input = kv_lora[..., dkv:]

    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    cos_q = cos_table[query_pos].view(d_rope).contiguous()
    sin_q = sin_table[query_pos].view(d_rope).contiguous()
    rope_inplace_query(q_rope, cos_q, sin_q)

    cos_k = cos_table[:kv_len]
    sin_k = sin_table[:kv_len]
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k

    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
    wK = wUKV_view[:, :d_nope, :]   # [nh, d_nope, dkv]
    wV = wUKV_view[:, d_nope:, :]   # [nh, dv, dkv]

    # Absorb wK into query: q_nope_latent[b,h,:] = q_nope[b,h,:] @ wK[h,:,:] -- shape [bs, nh, dkv]
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)
    # q_rope already has RoPE applied: [bs, nh, d_rope]
    # kv_nope_input: [bs, kv_len, dkv], k_rope: [bs, kv_len, d_rope]

    scale = 1.0 / math.sqrt(d_nope + d_rope)

    BLOCK_DKV = 512   # dkv=512, must be power of 2
    BLOCK_DROPE = 64  # d_rope=64
    BLOCK_KV = 64     # tile size over sequence length

    # Ensure contiguous
    q_nope_latent = q_nope_latent.contiguous()
    q_rope_c = q_rope.contiguous()
    kv_nope_c = kv_nope_input.contiguous()
    k_rope_c = k_rope.contiguous()

    M = torch.empty(bs, nh, dkv, dtype=torch.bfloat16, device=x.device)

    grid = (bs, nh)
    flash_decode_latent_kernel[grid](
        q_nope_latent, q_rope_c,
        kv_nope_c, k_rope_c,
        M,
        q_nope_latent.stride(0), q_nope_latent.stride(1), q_nope_latent.stride(2),
        q_rope_c.stride(0), q_rope_c.stride(1), q_rope_c.stride(2),
        kv_nope_c.stride(0), kv_nope_c.stride(1), kv_nope_c.stride(2),
        k_rope_c.stride(0), k_rope_c.stride(1), k_rope_c.stride(2),
        M.stride(0), M.stride(1), M.stride(2),
        kv_len=kv_len,
        DKV=dkv,
        DROPE=d_rope,
        scale=scale,
        BLOCK_KV=BLOCK_KV,
        BLOCK_DKV=BLOCK_DKV,
        BLOCK_DROPE=BLOCK_DROPE,
        num_warps=8,
    )

    # Apply wV: y_head[b,h,:] = M[b,h,:] @ wV[h,:,:].T  -> [bs, nh, dv]
    # M: [bs, nh, dkv], wV: [nh, dv, dkv] -> wV_T: [nh, dkv, dv]
    wV_T = wV.permute(0, 2, 1)   # [nh, dkv, dv]
    y_head = torch.einsum('bhd,hdk->bhk', M.to(wV.dtype), wV_T)

    y = y_head.reshape(bs, nh * dv)
    y = y.unsqueeze(1)
    output = F.linear(y, wO)

    return output, kv_cache.data
# EVOLVE-BLOCK-END
