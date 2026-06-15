# EVOLVE-BLOCK-START
"""
MLA Decode submission — FlashAttention via torch SDPA + absorbed KV trick.

Key changes vs baseline:
- Replace score/softmax/V-matmul chain with torch.nn.functional.scaled_dot_product_attention
  (uses cuDNN FlashAttention on H200, fused QK^T+softmax+V in SRAM)
- Construct composite Q = [q_nope_latent | q_rope] and K = [kv_nope | k_rope] so that
  the full dot product is computed in one SDPA call
- V for SDPA is just kv_nope (latent), then apply wV projection afterwards
- Ensure contiguous tensors for all large GEMMs
"""

import os
import math
from typing import Tuple
import torch
import torch.nn.functional as F
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


def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward step of the Multi-head Latent Attention (MLA) module using SDPA FlashAttention.
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
    q_lora = F.linear(x, wDQ)                   # [bs, 1, dq]
    kv_lora_input = F.linear(x, wDKV)           # [bs, 1, dkv+d_rope]

    kv_lora, kv_len = kv_cache(kv_lora_input)   # [bs, kv_len, dkv+d_rope]
    query_pos = kv_len - 1

    # Up-project queries
    q_up = F.linear(q_lora.squeeze(1), wUQ)     # [bs, nh*(d_nope+d_rope)]
    q_up = q_up.view(bs, nh, d_nope + d_rope)
    q_nope = q_up[..., :d_nope]                 # [bs, nh, d_nope]
    q_rope = q_up[..., d_nope:]                 # [bs, nh, d_rope]

    # Split KV latent
    kv_nope_input = kv_lora[..., :dkv].contiguous()   # [bs, kv_len, dkv]
    k_rope_input = kv_lora[..., dkv:]                  # [bs, kv_len, d_rope]

    # RoPE for keys and query
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)
    cos_q = cos_table[query_pos]  # [d_rope]
    sin_q = sin_table[query_pos]  # [d_rope]
    q_rope = q_rope * cos_q + _rotate_half(q_rope) * sin_q

    cos_k = cos_table[:kv_len]   # [kv_len, d_rope]
    sin_k = sin_table[:kv_len]   # [kv_len, d_rope]
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k  # [bs, kv_len, d_rope]

    # Absorbed KV: compute q_nope_latent = q_nope @ wK  -> [bs, nh, dkv]
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
    wK = wUKV_view[:, :d_nope, :].contiguous()          # [nh, d_nope, dkv]
    # Efficient batched matmul: q_nope [bs, nh, d_nope] x wK [nh, d_nope, dkv]
    # -> [bs, nh, dkv]
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)

    # Composite Q: [q_nope_latent | q_rope] -> [bs, nh, dkv+d_rope]
    # Composite K: [kv_nope | k_rope] -> [bs, kv_len, dkv+d_rope]
    # SDPA expects [bs, nh, sq, hd] for Q and [bs, nh, kv_len, hd] for K/V

    # Q: [bs, nh, 1, dkv+d_rope]
    Q_comp = torch.cat([q_nope_latent, q_rope], dim=-1).unsqueeze(2)  # [bs, nh, 1, dkv+d_rope]

    # K: [bs, nh, kv_len, dkv+d_rope]
    K_comp = torch.cat([kv_nope_input, k_rope], dim=-1)               # [bs, kv_len, dkv+d_rope]
    K_comp = K_comp.unsqueeze(1).expand(-1, nh, -1, -1)               # [bs, nh, kv_len, dkv+d_rope]

    # V: [bs, nh, kv_len, dkv] — use kv_nope latent, apply wV afterwards
    V_comp = kv_nope_input.unsqueeze(1).expand(-1, nh, -1, -1)        # [bs, nh, kv_len, dkv]

    scale = 1.0 / math.sqrt(d_nope + d_rope)

    # FlashAttention-style fused attention via SDPA
    # Q_comp, K_comp already contiguous after cat/unsqueeze
    M = F.scaled_dot_product_attention(
        Q_comp, K_comp, V_comp,
        scale=scale,
        is_causal=False,
    )  # [bs, nh, 1, dkv]
    M = M.squeeze(2)  # [bs, nh, dkv]

    # Apply wV projection: [nh, dkv, dv]
    wV = wUKV_view[:, d_nope:, :].contiguous()  # [nh, dv, dkv]
    wV_T = wV.permute(0, 2, 1).contiguous()     # [nh, dkv, dv]
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)  # [bs, nh, dv]

    y = y_head.reshape(bs, nh * dv)
    y = y.unsqueeze(1)
    output = F.linear(y, wO)

    return output, kv_cache.data
# EVOLVE-BLOCK-END
