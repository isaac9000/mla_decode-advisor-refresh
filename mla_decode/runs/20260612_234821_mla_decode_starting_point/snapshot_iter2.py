# EVOLVE-BLOCK-START
"""
MLA Decode with fused Triton flash-attention kernel.
Fuses nope+rope score computation, online softmax, and output accumulation
into a single kernel to eliminate writing O(bs*nh*kv_len) score matrix to HBM.
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
def _mla_flash_attn_kernel(
    # Query inputs
    q_nope_latent_ptr,  # [bs, nh, dkv]
    q_rope_ptr,         # [bs, nh, d_rope]
    # KV cache inputs
    kv_nope_ptr,        # [bs, kv_len, dkv]
    k_rope_ptr,         # [bs, kv_len, d_rope]
    # Output
    out_ptr,            # [bs, nh, dkv]
    # Strides for q_nope_latent
    stride_qnl_b, stride_qnl_h, stride_qnl_k,
    # Strides for q_rope
    stride_qr_b, stride_qr_h, stride_qr_d,
    # Strides for kv_nope
    stride_kvn_b, stride_kvn_s, stride_kvn_k,
    # Strides for k_rope
    stride_kr_b, stride_kr_s, stride_kr_d,
    # Strides for output
    stride_out_b, stride_out_h, stride_out_k,
    # Dimensions
    nh,
    kv_len,
    dkv: tl.constexpr,
    d_rope: tl.constexpr,
    scale,
    # Block sizes
    BLOCK_KV: tl.constexpr,
):
    """
    One program per (batch, head). Grid: (bs * nh,)
    Computes fused flash-attention with online softmax.
    """
    bh_idx = tl.program_id(0)
    b = bh_idx // nh
    h = bh_idx % nh

    # Load q_nope_latent[b, h, :] into registers
    q_nope_base = q_nope_latent_ptr + b * stride_qnl_b + h * stride_qnl_h
    q_nope_offs = tl.arange(0, dkv)
    q_nope = tl.load(q_nope_base + q_nope_offs * stride_qnl_k).to(tl.float32)

    # Load q_rope[b, h, :] into registers
    q_rope_base = q_rope_ptr + b * stride_qr_b + h * stride_qr_h
    q_rope_offs = tl.arange(0, d_rope)
    q_rope = tl.load(q_rope_base + q_rope_offs * stride_qr_d).to(tl.float32)

    # Pointers to kv_nope[b, :, :] and k_rope[b, :, :]
    kv_nope_base = kv_nope_ptr + b * stride_kvn_b
    k_rope_base = k_rope_ptr + b * stride_kr_b

    # Online softmax state
    m_prev = tl.full([1], float('-inf'), dtype=tl.float32)
    l_prev = tl.full([1], 0.0, dtype=tl.float32)
    acc = tl.zeros([dkv], dtype=tl.float32)

    kv_offs = tl.arange(0, dkv)
    kr_offs = tl.arange(0, d_rope)

    for start_s in range(0, kv_len, BLOCK_KV):
        s_offs = start_s + tl.arange(0, BLOCK_KV)
        s_mask = s_offs < kv_len

        # Load kv_nope tile: [BLOCK_KV, dkv]
        kv_tile_ptrs = kv_nope_base + s_offs[:, None] * stride_kvn_s + kv_offs[None, :] * stride_kvn_k
        kv_tile = tl.load(kv_tile_ptrs, mask=s_mask[:, None], other=0.0).to(tl.float32)

        # Load k_rope tile: [BLOCK_KV, d_rope]
        kr_tile_ptrs = k_rope_base + s_offs[:, None] * stride_kr_s + kr_offs[None, :] * stride_kr_d
        kr_tile = tl.load(kr_tile_ptrs, mask=s_mask[:, None], other=0.0).to(tl.float32)

        # Compute scores: q_nope [dkv] dot kv_tile [BLOCK_KV, dkv] -> [BLOCK_KV]
        scores_nope = tl.sum(kv_tile * q_nope[None, :], axis=1)  # [BLOCK_KV]
        scores_rope = tl.sum(kr_tile * q_rope[None, :], axis=1)  # [BLOCK_KV]
        scores = (scores_nope + scores_rope) * scale

        # Mask out-of-bounds
        scores = tl.where(s_mask, scores, float('-inf'))

        # Online softmax update
        m_curr = tl.max(scores, axis=0)
        m_new = tl.maximum(m_prev, m_curr)

        # Rescale previous accumulator
        alpha = tl.exp(m_prev - m_new)
        # Compute exp of current scores
        p = tl.exp(scores - m_new)
        # Mask p for out-of-bounds
        p = tl.where(s_mask, p, 0.0)

        l_new = alpha * l_prev + tl.sum(p, axis=0)

        # Update accumulator: rescale + add weighted kv
        acc = alpha * acc + tl.sum(p[:, None] * kv_tile, axis=0)

        m_prev = m_new
        l_prev = l_new

    # Normalize
    acc = acc / l_prev

    # Write output
    out_base = out_ptr + b * stride_out_b + h * stride_out_h
    tl.store(out_base + kv_offs * stride_out_k, acc.to(tl.bfloat16))


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


def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward step of the Multi-head Latent Attention (MLA) module.
    Uses fused Triton flash-attention kernel for the attention phase.
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

    # LoRA projections
    q_lora = F.linear(x, wDQ)
    kv_lora_input = F.linear(x, wDKV)

    # Update KV cache
    kv_lora, kv_len = kv_cache(kv_lora_input)
    query_pos = kv_len - 1

    # Up-project queries
    q_up = F.linear(q_lora.squeeze(1), wUQ)
    q_up = q_up.view(bs, nh, d_nope + d_rope)
    q_nope = q_up[..., :d_nope]
    q_rope = q_up[..., d_nope:]

    # Split KV cache into latent and rope parts
    kv_nope_input = kv_lora[..., :dkv]   # [bs, kv_len, dkv]
    k_rope_input = kv_lora[..., dkv:]    # [bs, kv_len, d_rope]

    # Apply RoPE to queries
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)
    cos_q = cos_table[query_pos]  # [d_rope]
    sin_q = sin_table[query_pos]  # [d_rope]
    q_rope = q_rope * cos_q + _rotate_half(q_rope) * sin_q

    # Apply RoPE to keys
    cos_k = cos_table[:kv_len]  # [kv_len, d_rope]
    sin_k = sin_table[:kv_len]  # [kv_len, d_rope]
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k  # [bs, kv_len, d_rope]

    # Absorb wK into q_nope to get q_nope_latent
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
    wK = wUKV_view[:, :d_nope, :]  # [nh, d_nope, dkv]
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)  # [bs, nh, dkv]

    # Make tensors contiguous for the Triton kernel
    q_nope_latent = q_nope_latent.contiguous()
    q_rope = q_rope.contiguous()
    kv_nope_input = kv_nope_input.contiguous()
    k_rope = k_rope.contiguous()

    scale = 1.0 / math.sqrt(d_nope + d_rope)

    # Allocate output: [bs, nh, dkv] - attention-weighted sum of kv_nope
    M = torch.empty(bs, nh, dkv, dtype=torch.bfloat16, device=x.device)

    # BLOCK_KV: tile size over sequence length
    BLOCK_KV = 64

    grid = (bs * nh,)
    _mla_flash_attn_kernel[grid](
        q_nope_latent, q_rope,
        kv_nope_input, k_rope,
        M,
        # strides q_nope_latent
        q_nope_latent.stride(0), q_nope_latent.stride(1), q_nope_latent.stride(2),
        # strides q_rope
        q_rope.stride(0), q_rope.stride(1), q_rope.stride(2),
        # strides kv_nope
        kv_nope_input.stride(0), kv_nope_input.stride(1), kv_nope_input.stride(2),
        # strides k_rope
        k_rope.stride(0), k_rope.stride(1), k_rope.stride(2),
        # strides output
        M.stride(0), M.stride(1), M.stride(2),
        # dims
        nh,
        kv_len,
        dkv=dkv,
        d_rope=d_rope,
        scale=scale,
        BLOCK_KV=BLOCK_KV,
        num_warps=4,
    )

    # Project output through wV
    wV = wUKV_view[:, d_nope:, :]  # [nh, dv, dkv]
    wV_T = wV.permute(0, 2, 1)     # [nh, dkv, dv]
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)  # [bs, nh, dv]

    y = y_head.reshape(bs, nh * dv)
    y = y.unsqueeze(1)
    output = F.linear(y, wO)

    return output, kv_cache.data
# EVOLVE-BLOCK-END
