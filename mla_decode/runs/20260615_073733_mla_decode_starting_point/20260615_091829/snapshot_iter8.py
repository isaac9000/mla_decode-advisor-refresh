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


@triton.jit
def _fused_add_scale_softmax_kernel(
    out_ptr, a_ptr, b_ptr,
    stride_out, stride_a, stride_b,
    n_cols, scale,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused: softmax((a + b) * scale) without materializing (a+b) to HBM.
    3-pass but reads a+b only twice (not three times for separate add then softmax).
    Saves one 134MB write + read vs baseline's separate add then softmax.
    """
    row = tl.program_id(0)
    row_off_a   = row * stride_a
    row_off_b   = row * stride_b
    row_off_out = row * stride_out
    col = tl.arange(0, BLOCK_SIZE)

    # Pass 1: compute max of (a+b)*scale
    max_val = tl.full([BLOCK_SIZE], float('-inf'), tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        va = tl.load(a_ptr + row_off_a + cur, mask=mask, other=float('-inf')).to(tl.float32)
        vb = tl.load(b_ptr + row_off_b + cur, mask=mask, other=0.0).to(tl.float32)
        max_val = tl.maximum(max_val, tl.where(mask, (va + vb) * scale, float('-inf')))
    row_max = tl.max(max_val)

    # Pass 2: compute exp and sum, store unnormalized exp
    sum_val = tl.full([BLOCK_SIZE], 0.0, tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        va = tl.load(a_ptr + row_off_a + cur, mask=mask, other=float('-inf')).to(tl.float32)
        vb = tl.load(b_ptr + row_off_b + cur, mask=mask, other=0.0).to(tl.float32)
        exp_val = tl.exp(tl.where(mask, (va + vb) * scale, float('-inf')) - row_max)
        tl.store(out_ptr + row_off_out + cur, exp_val.to(tl.bfloat16), mask=mask)
        sum_val += exp_val
    row_sum = tl.sum(sum_val)

    # Pass 3: normalize
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row_off_out + cur, (val / row_sum).to(tl.bfloat16), mask=mask)


def _fused_add_scale_softmax(a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
    """softmax((a + b) * scale) — fused, never writes the intermediate sum to HBM."""
    assert a.is_cuda and a.dtype == torch.bfloat16
    assert a.shape == b.shape
    n_rows, n_cols = a.shape

    if n_cols <= 32:
        BLOCK_SIZE = 32
    elif n_cols <= 64:
        BLOCK_SIZE = 64
    elif n_cols <= 128:
        BLOCK_SIZE = 128
    else:
        BLOCK_SIZE = 1 << (n_cols - 1).bit_length()
        BLOCK_SIZE = min(BLOCK_SIZE, 1024)

    # More warps = better warp-level parallelism for large rows
    num_warps = 8 if BLOCK_SIZE >= 512 else 4

    out = torch.empty_like(a)
    _fused_add_scale_softmax_kernel[(n_rows,)](
        out, a, b,
        out.stride(0), a.stride(0), b.stride(0),
        n_cols, scale,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out


@triton.jit
def _weighted_sum_kernel(
    # attn: [bs*nh, kv_len] bf16 (softmax output)
    # kv:   [bs, kv_len, DKV] bf16
    # out:  [bs*nh, DKV] bf16
    attn_ptr, kv_ptr, out_ptr,
    stride_attn,          # attn row stride (= kv_len)
    stride_kv_b,          # kv batch stride (= kv_len * DKV)
    stride_kv_t,          # kv token stride (= DKV)
    stride_out,           # out row stride (= DKV)
    kv_len,
    NH,                   # n_heads (to decode b and h from program_id)
    DKV: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """
    Fused weighted sum: out[b*nh+h] = sum_t(attn[b*nh+h, t] * kv[b, t, :])
    Each CTA handles one (batch, head) pair.
    Accumulates [DKV] fp32 — with 128 threads (num_warps=4), each thread holds DKV/128=4 values.
    """
    row = tl.program_id(0)
    b = row // NH
    h = row - b * NH   # h is unused but needed to map row -> b

    attn_base = attn_ptr + row * stride_attn
    kv_base = kv_ptr + b * stride_kv_b
    out_base = out_ptr + row * stride_out

    acc = tl.zeros([DKV], dtype=tl.float32)
    t_off = tl.arange(0, BLOCK_T)
    d_off = tl.arange(0, DKV)

    for t_start in range(0, kv_len, BLOCK_T):
        t = t_start + t_off
        mask_t = t < kv_len

        # Load attn weights: [BLOCK_T]
        w = tl.load(attn_base + t, mask=mask_t, other=0.0).to(tl.float32)

        # Load kv tile: [BLOCK_T, DKV]
        kv_tile = tl.load(
            kv_base + t[:, None] * stride_kv_t + d_off[None, :],
            mask=mask_t[:, None], other=0.0
        ).to(tl.float32)

        # Weighted sum: acc += sum_t(w[t] * kv_tile[t, :])
        acc += tl.sum(w[:, None] * kv_tile, axis=0)

    tl.store(out_base + d_off, acc.to(tl.bfloat16))


@triton.jit
def _attn_weighted_sum_dkv_tiled_kernel(
    # attn:    [bs*nh, kv_len]   bf16 — flattened softmax weights
    # kv_nope: [bs, kv_len, DKV] bf16
    # out:     [bs*nh, DKV]      bf16
    attn_ptr, kv_ptr, out_ptr,
    stride_attn,          # = kv_len
    stride_kv_b,          # = kv_len * DKV
    stride_kv_t,          # = DKV
    stride_out,           # = DKV
    kv_len,
    NH,
    DKV,
    TILE_DKV: tl.constexpr,
    BLOCK_T:  tl.constexpr,
):
    """
    out[b*nh+h, :] = attn[b*nh+h, :] @ kv_nope[b, :, :]
    Grid: [bs * nh * (DKV // TILE_DKV)]
    Each CTA handles TILE_DKV output elements, streaming over kv_len tokens.
    Low register pressure: only TILE_DKV fp32 accumulators + BLOCK_T attn weights.
    """
    pid = tl.program_id(0)
    n_dkv_tiles = DKV // TILE_DKV
    bh_idx      = pid // n_dkv_tiles
    dkv_tile    = pid  - bh_idx * n_dkv_tiles

    b = bh_idx // NH
    # h = bh_idx % NH   (unused for kv_nope indexing)

    d_off = dkv_tile * TILE_DKV + tl.arange(0, TILE_DKV)

    attn_base = attn_ptr + bh_idx * stride_attn
    kv_base   = kv_ptr   + b * stride_kv_b
    out_base  = out_ptr  + bh_idx * stride_out

    acc = tl.zeros([TILE_DKV], tl.float32)
    t_off = tl.arange(0, BLOCK_T)

    for t_start in range(0, kv_len, BLOCK_T):
        t = t_start + t_off
        mask_t = t < kv_len

        # attn weights [BLOCK_T]
        w = tl.load(attn_base + t, mask=mask_t, other=0.0).to(tl.float32)

        # kv tile [BLOCK_T, TILE_DKV]
        kv_tile = tl.load(
            kv_base + t[:, None] * stride_kv_t + d_off[None, :],
            mask=mask_t[:, None], other=0.0
        ).to(tl.float32)

        acc += tl.sum(w[:, None] * kv_tile, axis=0)

    tl.store(out_base + d_off, acc.to(tl.bfloat16))


def _attn_weighted_sum_tiled(
    attn: torch.Tensor,       # [bs, nh, kv_len] bf16
    kv_nope: torch.Tensor,    # [bs, kv_len, DKV] bf16
) -> torch.Tensor:
    """Tiled GEMV: M = attn @ kv_nope, tiled over DKV to reduce register pressure."""
    bs, nh, kv_len = attn.shape
    dkv = kv_nope.shape[2]

    TILE_DKV = 64   # 64 fp32 accumulators per CTA — low register pressure
    BLOCK_T  = 64   # token tile size

    assert dkv % TILE_DKV == 0
    n_dkv_tiles = dkv // TILE_DKV

    attn_flat = attn.reshape(bs * nh, kv_len).contiguous()
    kv_nope_c = kv_nope.contiguous()
    out = torch.empty(bs * nh, dkv, dtype=torch.bfloat16, device=attn.device)

    grid = (bs * nh * n_dkv_tiles,)
    _attn_weighted_sum_dkv_tiled_kernel[grid](
        attn_flat, kv_nope_c, out,
        attn_flat.stride(0),
        kv_nope_c.stride(0), kv_nope_c.stride(1),
        out.stride(0),
        kv_len, nh, dkv,
        TILE_DKV=TILE_DKV,
        BLOCK_T=BLOCK_T,
        num_warps=4,
    )
    return out.view(bs, nh, dkv)


@triton.jit
def _mla_flash_attn_kernel(
    # Inputs
    q_nope_lat_ptr,   # [bs, nh, dkv]  — q_nope already projected to latent space
    q_rope_ptr,       # [bs, nh, d_rope]
    kv_nope_ptr,      # [bs, kv_len, dkv]
    k_rope_in_ptr,    # [bs, kv_len, d_rope]  — pre-RoPE key rope input
    cos_ptr,          # [kv_len, d_rope]
    sin_ptr,          # [kv_len, d_rope]
    # Output
    out_ptr,          # [bs, nh, dkv]  — M = sum_t(attn_t * kv_nope[t])
    # Strides
    stride_qn_b, stride_qn_h,          # q_nope_lat strides (last dim = dkv, contiguous)
    stride_qr_b, stride_qr_h,          # q_rope strides
    stride_kv_b, stride_kv_t,          # kv_nope strides
    stride_kr_b, stride_kr_t,          # k_rope_in strides
    stride_cos_t,                       # cos/sin token stride (= d_rope)
    stride_out_b, stride_out_h,        # out strides
    # Sizes
    kv_len,
    scale,
    NH,
    # Constexpr tile sizes
    DKV: tl.constexpr,        # kv_lora_rank = 512
    D_ROPE: tl.constexpr,     # qk_rope_head_dim = 64
    D_ROPE_H: tl.constexpr,   # D_ROPE // 2 = 32
    BLOCK_T: tl.constexpr,    # tokens per tile
):
    """
    Flash-attention-style fused MLA kernel.
    Each CTA handles one (batch=b, head=h) pair.
    Streams kv_nope tiles once, computing scores + online-softmax + weighted sum.
    RoPE is applied inline to k_rope during the streaming pass.
    """
    pid = tl.program_id(0)
    b = pid // NH
    h = pid - b * NH

    # Base pointers for this (b, h)
    q_nope_base = q_nope_lat_ptr + b * stride_qn_b + h * stride_qn_h
    q_rope_base  = q_rope_ptr    + b * stride_qr_b + h * stride_qr_h
    kv_base      = kv_nope_ptr   + b * stride_kv_b
    kr_base      = k_rope_in_ptr + b * stride_kr_b
    out_base     = out_ptr       + b * stride_out_b + h * stride_out_h

    d_off   = tl.arange(0, DKV)
    dr0_off = tl.arange(0, D_ROPE_H)         # first half of rope dim
    dr1_off = D_ROPE_H + tl.arange(0, D_ROPE_H)  # second half of rope dim

    # Load q vectors into registers (fp32)
    q_nope = tl.load(q_nope_base + d_off).to(tl.float32)       # [DKV]
    q_r0   = tl.load(q_rope_base + dr0_off).to(tl.float32)     # [D_ROPE_H] first half
    q_r1   = tl.load(q_rope_base + dr1_off).to(tl.float32)     # [D_ROPE_H] second half

    # Online softmax state
    m_i = tl.full([1], float('-inf'), tl.float32)
    l_i = tl.zeros([1], tl.float32)
    acc  = tl.zeros([DKV], tl.float32)

    t_off = tl.arange(0, BLOCK_T)

    for t_start in range(0, kv_len, BLOCK_T):
        t = t_start + t_off
        mask_t = t < kv_len

        # --- Load kv_nope tile [BLOCK_T, DKV] ---
        kv_tile = tl.load(
            kv_base + t[:, None] * stride_kv_t + d_off[None, :],
            mask=mask_t[:, None], other=0.0
        ).to(tl.float32)

        # --- Nope scores: dot(q_nope, kv_nope[t]) ---
        score_nope = tl.sum(q_nope[None, :] * kv_tile, axis=1)  # [BLOCK_T]

        # --- Load k_rope_input halves [BLOCK_T, D_ROPE_H] each ---
        kr0 = tl.load(
            kr_base + t[:, None] * stride_kr_t + dr0_off[None, :],
            mask=mask_t[:, None], other=0.0
        ).to(tl.float32)
        kr1 = tl.load(
            kr_base + t[:, None] * stride_kr_t + dr1_off[None, :],
            mask=mask_t[:, None], other=0.0
        ).to(tl.float32)

        # --- Load cos/sin halves ---
        cos0 = tl.load(cos_ptr + t[:, None] * stride_cos_t + dr0_off[None, :],
                       mask=mask_t[:, None], other=1.0).to(tl.float32)
        sin0 = tl.load(sin_ptr + t[:, None] * stride_cos_t + dr0_off[None, :],
                       mask=mask_t[:, None], other=0.0).to(tl.float32)
        cos1 = tl.load(cos_ptr + t[:, None] * stride_cos_t + dr1_off[None, :],
                       mask=mask_t[:, None], other=1.0).to(tl.float32)
        sin1 = tl.load(sin_ptr + t[:, None] * stride_cos_t + dr1_off[None, :],
                       mask=mask_t[:, None], other=0.0).to(tl.float32)

        # rotate_half: [-kr1, kr0]
        # k_rope = kr * cos + rotate_half(kr) * sin
        # first half:  kr0 * cos0 + (-kr1) * sin0
        # second half: kr1 * cos1 + kr0 * sin1
        k_r0 = kr0 * cos0 - kr1 * sin0  # [BLOCK_T, D_ROPE_H]
        k_r1 = kr1 * cos1 + kr0 * sin1  # [BLOCK_T, D_ROPE_H]

        # --- RoPE score: dot(q_rope, k_rope[t]) ---
        score_rope = tl.sum(q_r0[None, :] * k_r0, axis=1) + \
                     tl.sum(q_r1[None, :] * k_r1, axis=1)  # [BLOCK_T]

        # Combined scaled score
        score = (score_nope + score_rope) * scale
        score = tl.where(mask_t, score, float('-inf'))

        # Online softmax update (Flash Attention style)
        m_new = tl.maximum(m_i, tl.max(score))
        exp_score = tl.exp(score - m_new)          # [BLOCK_T]
        alpha = tl.exp(m_i - m_new)                # rescale previous state

        l_i = alpha * l_i + tl.sum(exp_score)
        acc  = alpha * acc + tl.sum(exp_score[:, None] * kv_tile, axis=0)
        m_i  = m_new

    # Final normalization
    acc = acc / l_i
    tl.store(out_base + d_off, acc.to(tl.bfloat16))


def _mla_flash_attn(
    q_nope_latent: torch.Tensor,  # [bs, nh, dkv]
    q_rope: torch.Tensor,         # [bs, nh, d_rope]
    kv_nope: torch.Tensor,        # [bs, kv_len, dkv]
    k_rope_input: torch.Tensor,   # [bs, kv_len, d_rope]
    cos_table: torch.Tensor,      # [kv_len, d_rope]
    sin_table: torch.Tensor,      # [kv_len, d_rope]
    scale: float,
) -> torch.Tensor:
    bs, nh, dkv = q_nope_latent.shape
    kv_len = kv_nope.shape[1]
    d_rope = q_rope.shape[-1]

    # Ensure contiguous
    q_nope_latent = q_nope_latent.contiguous()
    q_rope = q_rope.contiguous()
    kv_nope = kv_nope.contiguous()
    k_rope_input = k_rope_input.contiguous()
    cos_table = cos_table.contiguous()
    sin_table = sin_table.contiguous()

    out = torch.empty(bs, nh, dkv, dtype=torch.bfloat16, device=q_nope_latent.device)

    BLOCK_T = 32  # tokens per tile

    grid = (bs * nh,)
    _mla_flash_attn_kernel[grid](
        q_nope_latent, q_rope, kv_nope, k_rope_input,
        cos_table, sin_table,
        out,
        q_nope_latent.stride(0), q_nope_latent.stride(1),
        q_rope.stride(0), q_rope.stride(1),
        kv_nope.stride(0), kv_nope.stride(1),
        k_rope_input.stride(0), k_rope_input.stride(1),
        cos_table.stride(0),
        out.stride(0), out.stride(1),
        kv_len, scale, nh,
        DKV=dkv,
        D_ROPE=d_rope,
        D_ROPE_H=d_rope // 2,
        BLOCK_T=BLOCK_T,
        num_warps=8,
    )
    return out


_weight_cache = {}


def _get_cached_weights(wUKV, wDQ, wDKV, nh, d_nope, dv, dkv):
    """
    Cache contiguous, pre-transposed weight tensors keyed by storage identity.
    Eliminates repeated slicing, viewing, and transposing on every forward call.
    Also caches wDQ_wDKV concatenation for a single fused GEMM.
    """
    key = (wUKV.data_ptr(), wDQ.data_ptr())
    if key not in _weight_cache:
        wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
        # wK: [nh, d_nope, dkv] -> make contiguous
        wK = wUKV_view[:, :d_nope, :].contiguous()
        # wV_T: [nh, dkv, dv] — pre-transposed for matmul
        wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1).contiguous()
        # Fused down-projection weight: cat([wDQ, wDKV], dim=0) -> [dq+dkv_in, dim]
        # F.linear(x, w) = x @ w.T, so concatenate along output dim (dim=0)
        wDQ_wDKV = torch.cat([wDQ, wDKV], dim=0).contiguous()
        _weight_cache[key] = (wK, wV_T, wDQ_wDKV)
    return _weight_cache[key]


def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised MLA decode — absorbed-key algorithm with cuBLAS GEMMs.
    Weight matrices are pre-cached as contiguous tensors; wDQ+wDKV batched into one GEMM.
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

    wDQ  = config.Q_proj_down_weight
    wDKV = config.KV_proj_down_weight
    wUQ  = config.Q_proj_up_weight
    wUKV = config.KV_proj_up_weight
    wO   = config.wo_weight

    wK, wV_T, wDQ_wDKV = _get_cached_weights(wUKV, wDQ, wDKV, nh, d_nope, dv, dkv)

    # Single fused GEMM for both down-projections (reads x once instead of twice)
    x_sq = x.squeeze(1)  # [bs, dim]
    q_kv_lora = F.linear(x_sq, wDQ_wDKV)  # [bs, dq + kv_lora_in]
    q_lora        = q_kv_lora[:, :dq].unsqueeze(1)   # [bs, 1, dq]
    kv_lora_input = q_kv_lora[:, dq:].unsqueeze(1)   # [bs, 1, kv_lora_in]

    kv_lora, kv_len = kv_cache(kv_lora_input)
    query_pos = kv_len - 1

    q_up = F.linear(q_lora.squeeze(1), wUQ)
    q_up = q_up.view(bs, nh, d_nope + d_rope)
    q_nope = q_up[..., :d_nope]
    q_rope = q_up[..., d_nope:]

    kv_nope_input = kv_lora[..., :dkv]
    k_rope_input  = kv_lora[..., dkv:]

    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    cos_q = cos_table[query_pos].view(d_rope).contiguous()
    sin_q = sin_table[query_pos].view(d_rope).contiguous()
    rope_inplace_query(q_rope, cos_q, sin_q)

    cos_k = cos_table[:kv_len]
    sin_k = sin_table[:kv_len]
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k

    # q_nope_latent: [bs, nh, dkv] = einsum('bhd,hdk->bhk', q_nope, wK)
    # wK is [nh, d_nope, dkv], contiguous — same einsum but now wK is contiguous
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)

    # Compute rope scores: [bs, nh, kv_len] — head-shared k_rope broadcast over nh heads
    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))

    scale = 1.0 / math.sqrt(d_nope + d_rope)

    # Use flash_attn_func with the nope latent as Q/K/V (GQA with 1 KV head, head_dim=dkv=512)
    # and scores_rope as additive attention bias.
    # flash_attn_func signature: (q, k, v, dropout_p, softmax_scale, causal, attn_bias, ...)
    # q: [bs, seqlen_q, nh, head_dim]
    # k: [bs, seqlen_k, nkv, head_dim]   (GQA: nkv=1, broadcast to nh)
    # v: [bs, seqlen_k, nkv, head_dim]
    # attn_bias: [bs, nh, seqlen_q, seqlen_k]  — add scores_rope here
    try:
        from flash_attn import flash_attn_func as _fa_func
        # q_fa: [bs, 1, nh, dkv]
        q_fa = q_nope_latent.unsqueeze(2).transpose(1, 2)  # [bs, 1, nh, dkv] -> actually need [bs, sq, nh, hd]
        # flash_attn expects [bs, seqlen, nheads, headdim]
        q_fa = q_nope_latent.unsqueeze(1)  # [bs, 1, nh, dkv]
        # GQA: k/v have 1 kv head
        k_fa = kv_nope_input.unsqueeze(2)   # [bs, kv_len, 1, dkv]
        v_fa = kv_nope_input.unsqueeze(2)   # [bs, kv_len, 1, dkv]
        # attn_bias: [bs, nh, 1, kv_len]
        attn_bias = (scores_rope * scale).unsqueeze(2)  # [bs, nh, 1, kv_len]
        # flash_attn output: [bs, 1, nh, dkv]
        M_fa = _fa_func(q_fa, k_fa, v_fa,
                        dropout_p=0.0,
                        softmax_scale=scale,
                        causal=False,
                        attn_bias=attn_bias)  # [bs, 1, nh, dkv]
        M = M_fa.squeeze(1)  # [bs, nh, dkv]
    except Exception:
        # Fallback: original cuBLAS path
        kv_nope_T = kv_nope_input.transpose(1, 2)
        scores_nope = torch.matmul(q_nope_latent, kv_nope_T)
        scores_nope_flat = scores_nope.reshape(bs * nh, kv_len)
        scores_rope_flat = scores_rope.reshape(bs * nh, kv_len)
        attn_flat = _fused_add_scale_softmax(scores_nope_flat, scores_rope_flat, scale)
        attn = attn_flat.view(bs, nh, kv_len)
        M = torch.matmul(attn, kv_nope_input)

    # y_head: [bs, nh, dv] = einsum('bhd,hdk->bhk', M, wV_T) — wV_T is [nh, dkv, dv], contiguous
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)

    y = y_head.reshape(bs, nh * dv)
    y = y.unsqueeze(1)
    output = F.linear(y, wO)

    return output, kv_cache.data
# EVOLVE-BLOCK-END
