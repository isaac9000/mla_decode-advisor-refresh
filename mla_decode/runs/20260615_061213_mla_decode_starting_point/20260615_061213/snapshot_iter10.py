# EVOLVE-BLOCK-START
"""
MLA Decode — fused RoPE+rope_scores Triton kernel eliminates k_rope materialization.

Key change vs baseline:
- Replace: k_rope = rotate(k_rope_input) then scores_rope = q_rope @ k_rope.T
- With: single Triton kernel that computes scores_rope[b,h,t] = q_rope[b,h,:] · rot(k_raw[b,t,:], t)
  without ever writing the full [bs, kv_len, 64] rotated k_rope tensor to HBM
- Saves ~100MB allocation + write bandwidth; 64-dim rope dot fits in registers
- Everything else identical to baseline (contiguous kv_nope, Triton softmax, einsum projections)
"""

import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config


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
def _rope_scores_kernel(
    # Inputs
    q_rope_ptr,     # [bs, nh, d_rope]  — rotated query rope
    k_raw_ptr,      # [bs, kv_len, d_rope]  — raw (un-rotated) key rope from KV cache
    cos_ptr,        # [kv_len, d_rope]
    sin_ptr,        # [kv_len, d_rope]
    # Output
    scores_ptr,     # [bs, nh, kv_len]
    # Strides
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_b, stride_k_t, stride_k_d,
    stride_cs_t, stride_cs_d,           # cos/sin share same strides
    stride_s_b, stride_s_h, stride_s_t,
    # Dims
    kv_len,
    NH: tl.constexpr,
    D_ROPE: tl.constexpr,   # 64
    BLOCK_T: tl.constexpr,  # tile over kv_len
):
    """
    One program = one (batch, head) pair.
    For each t in kv_len: apply RoPE to k_raw[b,t,:], dot with q_rope[b,h,:], write score.
    Avoids materializing rotated k_rope entirely.
    """
    pid = tl.program_id(0)
    b = pid // NH
    h = pid - b * NH

    # Load q_rope[b, h, :]: shape [D_ROPE]
    q_off = b * stride_q_b + h * stride_q_h
    q = tl.load(q_rope_ptr + q_off + tl.arange(0, D_ROPE) * stride_q_d).to(tl.float32)

    half = D_ROPE // 2
    d_idx = tl.arange(0, D_ROPE)

    # Loop over kv_len in tiles
    for start_t in range(0, kv_len, BLOCK_T):
        t_idx = start_t + tl.arange(0, BLOCK_T)
        mask_t = t_idx < kv_len

        # Load k_raw tile: [BLOCK_T, D_ROPE]
        k_ptrs = (k_raw_ptr + b * stride_k_b
                  + t_idx[:, None] * stride_k_t
                  + d_idx[None, :] * stride_k_d)
        k_raw_tile = tl.load(k_ptrs, mask=mask_t[:, None], other=0.0).to(tl.float32)

        # Load cos/sin tile: [BLOCK_T, D_ROPE]
        cos_tile = tl.load(
            cos_ptr + t_idx[:, None] * stride_cs_t + d_idx[None, :] * stride_cs_d,
            mask=mask_t[:, None], other=0.0).to(tl.float32)
        sin_tile = tl.load(
            sin_ptr + t_idx[:, None] * stride_cs_t + d_idx[None, :] * stride_cs_d,
            mask=mask_t[:, None], other=0.0).to(tl.float32)

        # Apply rotate_half: load first and second halves separately
        # For d < half: rot[t,d] = -k_raw[t, d+half]
        # For d >= half: rot[t,d] = k_raw[t, d-half]
        # Load k_raw again with shifted indices
        d_shifted = tl.where(d_idx < half, d_idx + half, d_idx - half)
        k_shift_ptrs = (k_raw_ptr + b * stride_k_b
                        + t_idx[:, None] * stride_k_t
                        + d_shifted[None, :] * stride_k_d)
        k_shifted = tl.load(k_shift_ptrs, mask=mask_t[:, None], other=0.0).to(tl.float32)
        k_neg_half = tl.where(d_idx[None, :] < half, -k_shifted, k_shifted)

        k_rot = k_raw_tile * cos_tile + k_neg_half * sin_tile  # [BLOCK_T, D_ROPE]

        # Dot product: scores[b, h, t] = sum_d q[d] * k_rot[t, d]
        scores_tile = tl.sum(q[None, :] * k_rot, axis=1)  # [BLOCK_T]

        # Store
        s_ptrs = scores_ptr + b * stride_s_b + h * stride_s_h + t_idx * stride_s_t
        tl.store(s_ptrs, scores_tile.to(tl.bfloat16), mask=mask_t)


def _rope_scores_triton(
    q_rope: torch.Tensor,    # [bs, nh, d_rope]  — already rotated
    k_raw: torch.Tensor,     # [bs, kv_len, d_rope]  — raw from KV cache
    cos_table: torch.Tensor, # [kv_len, d_rope]
    sin_table: torch.Tensor, # [kv_len, d_rope]
) -> torch.Tensor:
    bs, nh, d_rope = q_rope.shape
    kv_len = k_raw.shape[1]

    scores = torch.empty((bs, nh, kv_len), dtype=torch.bfloat16, device=q_rope.device)

    q_rope_c = q_rope.contiguous()
    k_raw_c = k_raw.contiguous()
    cos_c = cos_table.contiguous()
    sin_c = sin_table.contiguous()

    grid = (bs * nh,)
    BLOCK_T = 64
    _rope_scores_kernel[grid](
        q_rope_c, k_raw_c, cos_c, sin_c, scores,
        q_rope_c.stride(0), q_rope_c.stride(1), q_rope_c.stride(2),
        k_raw_c.stride(0), k_raw_c.stride(1), k_raw_c.stride(2),
        cos_c.stride(0), cos_c.stride(1),
        scores.stride(0), scores.stride(1), scores.stride(2),
        kv_len,
        NH=nh,
        D_ROPE=d_rope,
        BLOCK_T=BLOCK_T,
        num_warps=4,
        num_stages=2,
    )
    return scores


def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MLA decode: baseline structure + fused RoPE+dot Triton kernel for rope scores.
    Eliminates k_rope [bs, kv_len, 64] materialization.
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

    # Down-projection GEMMs
    q_lora = F.linear(x, wDQ)
    kv_lora_input = F.linear(x, wDKV)

    kv_lora, kv_len = kv_cache(kv_lora_input)   # [bs, kv_len, dkv+d_rope]
    query_pos = kv_len - 1

    # Up-project queries
    q_up = F.linear(q_lora.squeeze(1), wUQ)
    q_up = q_up.view(bs, nh, d_nope + d_rope)
    q_nope = q_up[..., :d_nope]                 # [bs, nh, d_nope]
    q_rope_raw = q_up[..., d_nope:]             # [bs, nh, d_rope]

    # KV slices — contiguous (faster per exp#8)
    kv_nope_input = kv_lora[..., :dkv].contiguous()  # [bs, kv_len, dkv]
    k_rope_raw = kv_lora[..., dkv:].contiguous()     # [bs, kv_len, d_rope]  (un-rotated)

    # RoPE tables
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # Apply RoPE to query only
    cos_q = cos_table[query_pos].view(d_rope).contiguous()
    sin_q = sin_table[query_pos].view(d_rope).contiguous()
    q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q  # [bs, nh, d_rope]

    # Absorbed KV: q_nope_latent = einsum('bhd,hdk->bhk', q_nope, wK) -> [bs, nh, dkv]
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
    wK = wUKV_view[:, :d_nope, :]
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)

    # Score nope via standard matmul (baseline)
    kv_nope_T = kv_nope_input.transpose(1, 2)                        # [bs, dkv, kv_len]
    scores_nope = torch.matmul(q_nope_latent, kv_nope_T)             # [bs, nh, kv_len]

    # Score rope via fused RoPE+dot kernel — no k_rope materialization
    cos_k = cos_table[:kv_len].contiguous()   # [kv_len, d_rope]
    sin_k = sin_table[:kv_len].contiguous()   # [kv_len, d_rope]
    scores_rope = _rope_scores_triton(q_rope, k_rope_raw, cos_k, sin_k)  # [bs, nh, kv_len]

    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale

    # Triton softmax (baseline)
    scores_flat = scores.reshape(bs * nh, kv_len)
    attn_flat = _triton_softmax(scores_flat)
    attn = attn_flat.view(bs, nh, kv_len)

    # V aggregation
    M = torch.matmul(attn, kv_nope_input)                             # [bs, nh, dkv]

    # Output projection (baseline)
    wV = wUKV_view[:, d_nope:, :]
    wV_T = wV.permute(0, 2, 1)
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)                   # [bs, nh, dv]

    y = y_head.reshape(bs, nh * dv).unsqueeze(1)
    output = F.linear(y, wO)

    return output, kv_cache.data
# EVOLVE-BLOCK-END
