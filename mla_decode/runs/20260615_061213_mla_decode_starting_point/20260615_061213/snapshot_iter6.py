# EVOLVE-BLOCK-START
"""
MLA Decode — xformers memory_efficient_attention with nope scores as attn_bias.

Strategy:
- Try xformers MEA: Q_rope [bs,1,nh,d_rope], K_rope [bs,kv_len,1,d_rope], V [bs,kv_len,1,dkv]
  with scores_nope [bs,nh,1,kv_len] as attn_bias (injected pre-softmax)
- xformers handles GQA (1 KV group) natively, d_rope=64 is within FlashAttention limits
- Fallback: baseline structure but with F.softmax (float32) replacing Triton softmax
"""

import math
from typing import Tuple
import torch
import torch.nn.functional as F
from reference import KVCache, Config

try:
    import xformers.ops as xops
    _XFORMERS = True
except ImportError:
    _XFORMERS = False

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
    MLA decode using xformers MEA (with nope bias) or baseline fallback.
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

    # Split KV latent
    kv_nope_input = kv_lora[..., :dkv].contiguous()  # [bs, kv_len, dkv]
    k_rope_input = kv_lora[..., dkv:]                 # [bs, kv_len, d_rope]

    # RoPE
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)
    cos_q = cos_table[query_pos]   # [d_rope]
    sin_q = sin_table[query_pos]   # [d_rope]
    q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q  # [bs, nh, d_rope]

    cos_k = cos_table[:kv_len]
    sin_k = sin_table[:kv_len]
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k  # [bs, kv_len, d_rope]

    # Absorbed KV: q_nope_latent = q_nope @ wK -> [bs, nh, dkv]
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
    wK = wUKV_view[:, :d_nope, :].contiguous()       # [nh, d_nope, dkv]
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)  # [bs, nh, dkv]

    # scores_nope: [bs, nh, 1, kv_len] — no per-head KV expansion, uses broadcast
    kv_nope_T = kv_nope_input.transpose(1, 2)        # [bs, dkv, kv_len]
    scores_nope = torch.matmul(q_nope_latent, kv_nope_T)  # [bs, nh, kv_len] via broadcast
    scale = 1.0 / math.sqrt(d_nope + d_rope)

    if _XFORMERS:
        # xformers MEA: Q [bs, sq=1, nh, d_rope], K [bs, kv_len, 1, d_rope], V [bs, kv_len, 1, dkv]
        # attn_bias: [bs, nh, 1, kv_len] — adds nope scores before softmax
        # scale applied to rope scores only; nope scores already folded in via bias
        Q_xf = q_rope.unsqueeze(2).transpose(1, 2)              # [bs, 1, nh, d_rope]
        K_xf = k_rope.unsqueeze(2)                              # [bs, kv_len, 1, d_rope]
        V_xf = kv_nope_input.unsqueeze(2)                       # [bs, kv_len, 1, dkv]

        # attn_bias adds nope contribution (pre-scaled) to rope scores
        # rope scale = 1/sqrt(d_nope+d_rope), nope scores already in correct units
        # We pass nope scores scaled by scale as bias; rope scores will be scaled by scale inside MEA
        # So: total_score = scale*(q_rope·k_rope) + scale*scores_nope_unscaled
        # = scale*(q_rope·k_rope + q_nope_latent·kv_nope^T)  ✓
        attn_bias = (scores_nope * scale).unsqueeze(2)  # [bs, nh, 1, kv_len]

        M_xf = xops.memory_efficient_attention(
            Q_xf, K_xf, V_xf,
            attn_bias=attn_bias,
            scale=scale,
        )  # [bs, 1, nh, dkv]
        M = M_xf.squeeze(1).transpose(1, 2)  # [bs, nh, dkv] -- wait, [bs,1,nh,dkv]->[bs,nh,dkv]
        # Actually xformers returns [bs, sq, nh, dkv] -> [bs, 1, nh, dkv]
        M = M_xf.view(bs, nh, dkv)
    else:
        # Fallback: baseline matmul path with F.softmax
        scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))  # [bs, nh, kv_len]
        scores = (scores_nope + scores_rope) * scale
        attn = F.softmax(scores.float(), dim=-1).to(torch.bfloat16)   # [bs, nh, kv_len]
        M = torch.matmul(attn, kv_nope_input)                         # [bs, nh, dkv]

    # Apply wV projection
    wV = wUKV_view[:, d_nope:, :].contiguous()       # [nh, dv, dkv]
    wV_T = wV.permute(0, 2, 1).contiguous()          # [nh, dkv, dv]
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)   # [bs, nh, dv]

    y = y_head.reshape(bs, nh * dv).unsqueeze(1)
    output = F.linear(y, wO)

    return output, kv_cache.data
# EVOLVE-BLOCK-END
