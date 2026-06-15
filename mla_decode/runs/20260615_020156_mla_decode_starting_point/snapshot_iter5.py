# EVOLVE-BLOCK-START
"""
MLA Decode submission — weight absorption/precomputation to reduce serial GEMMs.
"""

import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config

# Weight cache: keyed by id(config)
_weight_cache = {}


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


def _get_weight_cache(config):
    """
    Cache pre-transposed/contiguous weight tensors to avoid repeated reshape/permute ops.
    No large weight fusions — just ensuring the right shapes are ready.
    """
    key = id(config)
    if key in _weight_cache:
        return _weight_cache[key]

    nh = config.n_heads
    dkv = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv = config.v_head_dim

    wUKV = config.KV_proj_up_weight    # [nh*(d_nope+dv), dkv]

    wUKV_v = wUKV.view(nh, d_nope + dv, dkv)
    wK = wUKV_v[:, :d_nope, :].contiguous()  # [nh, d_nope, dkv]
    wV = wUKV_v[:, d_nope:, :].contiguous()  # [nh, dv,     dkv]

    # Pre-transpose wK for the bmm: q_nope [bs*nh, 1, d_nope] @ wK_T [bs*nh, d_nope, dkv]
    # wK_T[h] = wK[h].T = [dkv, d_nope].T — but we want [d_nope, dkv] so wK is already right.
    # For bmm: (bs*nh, 1, d_nope) @ (nh, d_nope, dkv) → need wK expanded to (bs*nh, d_nope, dkv)
    # Instead store wK as-is and expand at runtime — or store wK.permute(0,2,1) for the other direction.

    # For wV bmm: M [bs*nh, 1, dkv] @ wV_T [bs*nh, dkv, dv]
    # wV[h]: [dv, dkv] -> wV_T[h]: [dkv, dv]
    wV_T = wV.permute(0, 2, 1).contiguous()  # [nh, dkv, dv]

    cache = {
        'wK':   wK,    # [nh, d_nope, dkv]
        'wV_T': wV_T,  # [nh, dkv, dv]
    }
    _weight_cache[key] = cache
    return cache


def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MLA decode — baseline structure with einsum replaced by bmm and cached weight shapes.
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
    wO   = config.wo_weight

    # Get cached pre-shaped weights
    wcache = _get_weight_cache(config)
    wK   = wcache['wK']    # [nh, d_nope, dkv]  contiguous
    wV_T = wcache['wV_T']  # [nh, dkv,    dv]   contiguous

    x_sq = x.squeeze(1)  # [bs, dim]

    # --- Two-step Q projection (bandwidth-efficient) ---
    q_lora = F.linear(x_sq, wDQ)            # [bs, dq]
    q_up   = F.linear(q_lora, wUQ)          # [bs, nh*(d_nope+d_rope)]
    q_up   = q_up.view(bs, nh, d_nope + d_rope)
    q_nope = q_up[..., :d_nope]             # [bs, nh, d_nope]
    q_rope = q_up[..., d_nope:].contiguous() # [bs, nh, d_rope]

    # --- KV cache update ---
    kv_lora_input = F.linear(x, wDKV)       # [bs, 1, dkv+d_rope]
    kv_lora, kv_len = kv_cache(kv_lora_input)
    query_pos = kv_len - 1

    kv_nope_input = kv_lora[..., :dkv]      # [bs, kv_len, dkv]
    k_rope_input  = kv_lora[..., dkv:]      # [bs, kv_len, d_rope]

    # --- RoPE ---
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)
    cos_q = cos_table[query_pos].view(d_rope).contiguous()
    sin_q = sin_table[query_pos].view(d_rope).contiguous()
    rope_inplace_query(q_rope, cos_q, sin_q)

    cos_k = cos_table[:kv_len]
    sin_k = sin_table[:kv_len]
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k

    # --- q_nope_latent via bmm (replaces einsum 'bhd,hdk->bhk') ---
    # q_nope: [bs, nh, d_nope] -> [bs*nh, 1, d_nope]
    # wK:     [nh, d_nope, dkv] -> expand [bs, nh, d_nope, dkv] -> [bs*nh, d_nope, dkv]
    q_nope_3d = q_nope.reshape(bs * nh, 1, d_nope)
    wK_exp    = wK.unsqueeze(0).expand(bs, -1, -1, -1).reshape(bs * nh, d_nope, dkv)
    q_nope_latent = torch.bmm(q_nope_3d, wK_exp).view(bs, nh, dkv)  # [bs, nh, dkv]

    # --- Attention scores ---
    scale = 1.0 / math.sqrt(d_nope + d_rope)
    kv_nope_T   = kv_nope_input.transpose(1, 2)                      # [bs, dkv, kv_len]
    scores_nope = torch.matmul(q_nope_latent, kv_nope_T)             # [bs, nh, kv_len]
    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))     # [bs, nh, kv_len]
    scores      = (scores_nope + scores_rope) * scale

    scores_flat = scores.reshape(bs * nh, kv_len)
    attn_flat   = _triton_softmax(scores_flat)
    attn        = attn_flat.view(bs, nh, kv_len)

    # --- Weighted sum over kv_nope ---
    M = torch.matmul(attn, kv_nope_input)   # [bs, nh, dkv]

    # --- Output projection via bmm (replaces einsum 'bhd,hdk->bhk') ---
    # M:    [bs, nh, dkv]   -> [bs*nh, 1, dkv]
    # wV_T: [nh, dkv, dv]   -> expand [bs, nh, dkv, dv] -> [bs*nh, dkv, dv]
    M_3d     = M.reshape(bs * nh, 1, dkv)
    wV_T_exp = wV_T.unsqueeze(0).expand(bs, -1, -1, -1).reshape(bs * nh, dkv, dv)
    y_head   = torch.bmm(M_3d, wV_T_exp).view(bs, nh * dv)          # [bs, nh*dv]

    output = F.linear(y_head, wO).unsqueeze(1)                       # [bs, 1, dim_out]

    return output, kv_cache.data
# EVOLVE-BLOCK-END
