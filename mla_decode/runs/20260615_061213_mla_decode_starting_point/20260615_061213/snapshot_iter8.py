# EVOLVE-BLOCK-START
"""
MLA Decode — baseline structure, targeted improvements:
1. Remove kv_nope .contiguous() copy (non-contiguous slice works in matmul)
2. Remove k_rope_input .contiguous() copy
3. Replace 3-pass Triton softmax with F.softmax (fp32, native 3D, no reshape)
4. Keep proven einsum for weight projections
"""

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
    MLA decode: baseline structure with copy elimination and F.softmax.
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
    q_rope = q_up[..., d_nope:]                 # [bs, nh, d_rope]

    # Non-contiguous slices of KV cache — avoid .contiguous() copies
    # PyTorch matmul handles non-contiguous inputs correctly
    kv_nope_input = kv_lora[..., :dkv]          # [bs, kv_len, dkv]  (non-contiguous view)
    k_rope_input = kv_lora[..., dkv:]           # [bs, kv_len, d_rope] (non-contiguous view)

    # RoPE
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)
    cos_q = cos_table[query_pos].view(d_rope).contiguous()
    sin_q = sin_table[query_pos].view(d_rope).contiguous()
    q_rope = q_rope * cos_q + _rotate_half(q_rope) * sin_q

    cos_k = cos_table[:kv_len]
    sin_k = sin_table[:kv_len]
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k  # [bs, kv_len, d_rope]

    # Absorbed KV: q_nope_latent = einsum('bhd,hdk->bhk', q_nope, wK) -> [bs, nh, dkv]
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
    wK = wUKV_view[:, :d_nope, :]
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)

    # Score matmuls — separate (no per-head KV expansion, uses broadcast)
    kv_nope_T = kv_nope_input.transpose(1, 2)   # [bs, dkv, kv_len]
    scores_nope = torch.matmul(q_nope_latent, kv_nope_T)                # [bs, nh, kv_len]
    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))        # [bs, nh, kv_len]

    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale                        # [bs, nh, kv_len]

    # F.softmax: single-pass fused kernel, native 3D (no reshape needed)
    attn = F.softmax(scores.float(), dim=-1).to(torch.bfloat16)         # [bs, nh, kv_len]

    # V aggregation
    M = torch.matmul(attn, kv_nope_input)                               # [bs, nh, dkv]

    # wV projection
    wV = wUKV_view[:, d_nope:, :]
    wV_T = wV.permute(0, 2, 1)
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)                      # [bs, nh, dv]

    y = y_head.reshape(bs, nh * dv).unsqueeze(1)
    output = F.linear(y, wO)

    return output, kv_cache.data
# EVOLVE-BLOCK-END
