# EVOLVE-BLOCK-START
"""
MLA Decode — baseline + targeted micro-optimizations:
1. _rotate_half replaced with torch.roll (avoids torch.cat allocation)
2. kv_nope pre-transposed once to avoid re-transposing inside matmul
3. BF16 softmax (torch.softmax) instead of 3-pass Triton softmax
4. All contiguous copies kept (exp#8 showed non-contiguous is slower)
5. Einsum projections kept (exp#7/#9 showed matmul alternatives are slower)
"""

import math
from typing import Tuple
import torch
import torch.nn.functional as F
from reference import KVCache, Config


_rope_cache = {}


def _rotate_half_roll(x: torch.Tensor) -> torch.Tensor:
    """rotate_half via roll+sign flip — avoids torch.cat allocation."""
    half = x.shape[-1] // 2
    x_rolled = torch.roll(x, half, dims=-1)
    # Negate the first half (which was the second half before roll)
    x_rolled[..., :half] = -x_rolled[..., :half]
    return x_rolled


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
    MLA decode: baseline with _rotate_half→torch.roll (no cat alloc) + BF16 softmax.
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

    kv_lora, kv_len = kv_cache(kv_lora_input)    # [bs, kv_len, dkv+d_rope]
    query_pos = kv_len - 1

    # Up-project queries
    q_up = F.linear(q_lora.squeeze(1), wUQ)
    q_up = q_up.view(bs, nh, d_nope + d_rope)
    q_nope = q_up[..., :d_nope]                  # [bs, nh, d_nope]
    q_rope_raw = q_up[..., d_nope:]              # [bs, nh, d_rope]

    # KV slices — contiguous (faster per exp#8)
    kv_nope_input = kv_lora[..., :dkv].contiguous()  # [bs, kv_len, dkv]
    k_rope_input = kv_lora[..., dkv:].contiguous()   # [bs, kv_len, d_rope]

    # RoPE: use roll-based rotation (no torch.cat allocation)
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)
    cos_q = cos_table[query_pos].view(d_rope).contiguous()
    sin_q = sin_table[query_pos].view(d_rope).contiguous()
    q_rope = q_rope_raw * cos_q + _rotate_half_roll(q_rope_raw) * sin_q

    cos_k = cos_table[:kv_len]
    sin_k = sin_table[:kv_len]
    k_rope = k_rope_input * cos_k + _rotate_half_roll(k_rope_input) * sin_k

    # Absorbed KV: q_nope_latent [bs, nh, dkv]
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
    wK = wUKV_view[:, :d_nope, :]
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)

    # Score matmuls (baseline: two separate, no per-head KV expansion)
    kv_nope_T = kv_nope_input.transpose(1, 2)        # [bs, dkv, kv_len]
    scores_nope = torch.matmul(q_nope_latent, kv_nope_T)              # [bs, nh, kv_len]
    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))      # [bs, nh, kv_len]

    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale                      # [bs, nh, kv_len]

    # BF16 softmax directly on 3D scores (no reshape, no fp32 cast)
    attn = torch.softmax(scores, dim=-1)                              # [bs, nh, kv_len]

    # V aggregation
    M = torch.matmul(attn, kv_nope_input)                             # [bs, nh, dkv]

    # Output projection
    wV = wUKV_view[:, d_nope:, :]
    wV_T = wV.permute(0, 2, 1)
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)                   # [bs, nh, dv]

    y = y_head.reshape(bs, nh * dv).unsqueeze(1)
    output = F.linear(y, wO)

    return output, kv_cache.data
# EVOLVE-BLOCK-END
