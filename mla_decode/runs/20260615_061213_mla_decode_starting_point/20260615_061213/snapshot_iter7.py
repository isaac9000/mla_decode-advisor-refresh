# EVOLVE-BLOCK-START
"""
MLA Decode — optimized baseline: single fused score matmul, F.softmax, matmul replacing einsum.

Key changes:
1. Single fused score matmul: Q_combined=[q_nope_latent|q_rope] @ K_combined^T in one call
2. F.softmax (fp32) instead of 3-pass Triton softmax
3. torch.matmul replacing einsum for q_nope_latent and wV projections
4. Single contiguous K_combined = cat(kv_nope, k_rope) to avoid separate non-contiguous slices
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
    Optimized MLA decode: single score matmul, F.softmax, matmul-not-einsum.
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
    q_lora = F.linear(x, wDQ)                       # [bs, 1, dq]
    kv_lora_input = F.linear(x, wDKV)               # [bs, 1, dkv+d_rope]

    kv_lora, kv_len = kv_cache(kv_lora_input)        # [bs, kv_len, dkv+d_rope]
    query_pos = kv_len - 1

    # Up-project queries
    q_up = F.linear(q_lora.squeeze(1), wUQ)          # [bs, nh*(d_nope+d_rope)]
    q_up = q_up.view(bs, nh, d_nope + d_rope)
    q_nope = q_up[..., :d_nope]                      # [bs, nh, d_nope]
    q_rope_raw = q_up[..., d_nope:]                  # [bs, nh, d_rope]

    # Split KV latent — get contiguous slices for non-contiguous kv_lora
    kv_nope_input = kv_lora[..., :dkv].contiguous()  # [bs, kv_len, dkv]
    k_rope_input = kv_lora[..., dkv:].contiguous()   # [bs, kv_len, d_rope]

    # RoPE for query and keys
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)
    cos_q = cos_table[query_pos]   # [d_rope]
    sin_q = sin_table[query_pos]   # [d_rope]
    q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q  # [bs, nh, d_rope]

    cos_k = cos_table[:kv_len]
    sin_k = sin_table[:kv_len]
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k  # [bs, kv_len, d_rope]

    # Absorbed KV: q_nope_latent = q_nope @ wK -> [bs, nh, dkv]
    # einsum('bhd,hdk->bhk') is equivalent to:
    # [bs, nh, 1, d_nope] @ [1, nh, d_nope, dkv] -> [bs, nh, 1, dkv] -> squeeze
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
    wK = wUKV_view[:, :d_nope, :].contiguous()       # [nh, d_nope, dkv]
    # Correct broadcast: unsqueeze to [bs, nh, 1, d_nope] @ [nh, d_nope, dkv]
    q_nope_latent = torch.matmul(q_nope.unsqueeze(2), wK).squeeze(2)  # [bs, nh, dkv]

    # Build single contiguous combined Q and K for one fused score matmul
    # Q_combined: [bs, nh, dkv+d_rope]
    # K_combined: [bs, kv_len, dkv+d_rope] — single contiguous cat avoids two separate matmuls
    Q_combined = torch.cat([q_nope_latent, q_rope], dim=-1)          # [bs, nh, dkv+d_rope]
    K_combined = torch.cat([kv_nope_input, k_rope], dim=-1)          # [bs, kv_len, dkv+d_rope]

    # Single score matmul: [bs, nh, dkv+d_rope] @ [bs, dkv+d_rope, kv_len]
    # Uses PyTorch broadcast: [bs, nh, d] @ [bs, d, kv_len] -> [bs, nh, kv_len]
    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = torch.matmul(Q_combined, K_combined.transpose(-2, -1)) * scale  # [bs, nh, kv_len]

    # F.softmax in fp32 (cuDNN fused, faster than 3-pass Triton softmax)
    attn = F.softmax(scores.float(), dim=-1).to(torch.bfloat16)      # [bs, nh, kv_len]

    # Weighted V sum: [bs, nh, kv_len] @ [bs, kv_len, dkv] -> [bs, nh, dkv]
    M = torch.matmul(attn, kv_nope_input)                             # [bs, nh, dkv]

    # wV projection via matmul (replacing einsum)
    # M: [bs, nh, dkv], wV_T: [nh, dkv, dv] -> [bs, nh, dv]
    wV = wUKV_view[:, d_nope:, :].contiguous()        # [nh, dv, dkv]
    wV_T = wV.permute(0, 2, 1).contiguous()           # [nh, dkv, dv]
    # Correct broadcast: [bs, nh, 1, dkv] @ [nh, dkv, dv] -> [bs, nh, 1, dv] -> squeeze
    y_head = torch.matmul(M.unsqueeze(2), wV_T).squeeze(2)  # [bs, nh, dv]

    y = y_head.reshape(bs, nh * dv).unsqueeze(1)
    output = F.linear(y, wO)

    return output, kv_cache.data
# EVOLVE-BLOCK-END
