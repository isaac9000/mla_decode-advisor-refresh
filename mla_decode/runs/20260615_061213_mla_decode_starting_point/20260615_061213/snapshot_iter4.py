# EVOLVE-BLOCK-START
"""
MLA Decode — custom Triton fused attention kernel with online softmax.

Key design:
- Grid (bs * nh,): each program handles one (batch, head) pair
- Streams through kv_len in tiles (flash-attention style)
- Computes q_nope_latent · kv_nope^T + q_rope · k_rope^T per tile
- Maintains online softmax (running max/sum) across tiles
- Accumulates V=kv_nope output in SRAM — never materializes full score tensor
- K/V shared across all heads (1 KV group / MQA), each program reads from same base ptr
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
def _mla_fused_attn_kernel(
    # Query inputs (per-head)
    q_nope_latent_ptr,   # [bs, nh, dkv]
    q_rope_ptr,          # [bs, nh, d_rope]
    # Key/Value inputs (shared across heads — 1 KV group)
    kv_nope_ptr,         # [bs, kv_len, dkv]
    k_rope_ptr,          # [bs, kv_len, d_rope]
    # Output
    out_ptr,             # [bs, nh, dkv]
    # Strides
    stride_qn_b, stride_qn_h, stride_qn_d,   # q_nope_latent
    stride_qr_b, stride_qr_h, stride_qr_d,   # q_rope
    stride_kn_b, stride_kn_s, stride_kn_d,   # kv_nope
    stride_kr_b, stride_kr_s, stride_kr_d,   # k_rope
    stride_o_b,  stride_o_h,  stride_o_d,    # out
    # Dimensions
    kv_len,
    scale,
    DKV: tl.constexpr,      # kv_lora_rank = 512
    D_ROPE: tl.constexpr,   # qk_rope_head_dim = 64
    BLOCK_N: tl.constexpr,  # tile size along kv_len
):
    """
    One program = one (batch, head) pair.
    Streams over kv_len in blocks of BLOCK_N, maintaining online softmax state.
    """
    pid = tl.program_id(0)
    nh = tl.num_programs(0)  # won't use directly; compute b,h from pid
    # Actually we need total nh from outside — use pid directly
    # pid = b * NH + h
    # We'll pass NH via a separate arg but for simplicity use gridDim
    # Compute b and h from pid — NH is passed implicitly via num_programs
    # We pass NH as a constexpr
    NH: tl.constexpr  # will be passed
    b = pid // NH
    h = pid - b * NH

    # Load q_nope_latent for this (b, h): shape [DKV]
    q_nope_off = b * stride_qn_b + h * stride_qn_h
    q_nope = tl.load(
        q_nope_latent_ptr + q_nope_off + tl.arange(0, DKV) * stride_qn_d,
    ).to(tl.float32)

    # Load q_rope for this (b, h): shape [D_ROPE]
    q_rope_off = b * stride_qr_b + h * stride_qr_h
    q_rope = tl.load(
        q_rope_ptr + q_rope_off + tl.arange(0, D_ROPE) * stride_qr_d,
    ).to(tl.float32)

    # Online softmax state
    m_i = tl.full([1], -float("inf"), dtype=tl.float32)
    l_i = tl.full([1], 0.0, dtype=tl.float32)
    acc = tl.zeros([DKV], dtype=tl.float32)

    kv_nope_base = b * stride_kn_b
    k_rope_base = b * stride_kr_b

    for start_n in range(0, kv_len, BLOCK_N):
        n_off = start_n + tl.arange(0, BLOCK_N)
        mask_n = n_off < kv_len

        # Load kv_nope tile: [BLOCK_N, DKV]
        kv_nope_ptrs = kv_nope_ptr + kv_nope_base + n_off[:, None] * stride_kn_s + tl.arange(0, DKV)[None, :] * stride_kn_d
        kv_tile = tl.load(kv_nope_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # Load k_rope tile: [BLOCK_N, D_ROPE]
        k_rope_ptrs = k_rope_ptr + k_rope_base + n_off[:, None] * stride_kr_s + tl.arange(0, D_ROPE)[None, :] * stride_kr_d
        kr_tile = tl.load(k_rope_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # Compute scores: dot(q_nope, kv_tile^T) + dot(q_rope, kr_tile^T)
        # q_nope: [DKV], kv_tile: [BLOCK_N, DKV] -> scores_nope: [BLOCK_N]
        scores_nope = tl.sum(q_nope[None, :] * kv_tile, axis=1)
        scores_rope = tl.sum(q_rope[None, :] * kr_tile, axis=1)
        scores = (scores_nope + scores_rope) * scale

        # Mask padding positions
        scores = tl.where(mask_n, scores, -float("inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores))
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(scores - m_new)
        l_new = alpha * l_i + tl.sum(beta)

        # Accumulate weighted V (= kv_nope tile)
        acc = acc * alpha + tl.sum(beta[:, None] * kv_tile, axis=0)

        m_i = m_new
        l_i = l_new

    # Normalize
    acc = acc / l_i

    # Write output
    out_off = b * stride_o_b + h * stride_o_h
    tl.store(
        out_ptr + out_off + tl.arange(0, DKV) * stride_o_d,
        acc.to(tl.bfloat16),
    )


# We need NH as constexpr — wrap with a launcher that handles this
def _mla_fused_attn(
    q_nope_latent: torch.Tensor,  # [bs, nh, dkv]
    q_rope: torch.Tensor,          # [bs, nh, d_rope]
    kv_nope: torch.Tensor,         # [bs, kv_len, dkv]
    k_rope: torch.Tensor,          # [bs, kv_len, d_rope]
    scale: float,
    BLOCK_N: int = 64,
) -> torch.Tensor:
    bs, nh, dkv = q_nope_latent.shape
    _, _, d_rope = q_rope.shape
    kv_len = kv_nope.shape[1]

    out = torch.empty((bs, nh, dkv), dtype=torch.bfloat16, device=q_nope_latent.device)

    # Make inputs contiguous and float16/bfloat16
    q_nope_latent = q_nope_latent.contiguous()
    q_rope = q_rope.contiguous()
    kv_nope = kv_nope.contiguous()
    k_rope = k_rope.contiguous()

    grid = (bs * nh,)

    # Use a wrapper kernel that has NH as a runtime arg (not constexpr)
    # since nh=128 is fixed, we inline it as constexpr
    _mla_fused_attn_kernel_launcher[grid](
        q_nope_latent, q_rope, kv_nope, k_rope, out,
        q_nope_latent.stride(0), q_nope_latent.stride(1), q_nope_latent.stride(2),
        q_rope.stride(0), q_rope.stride(1), q_rope.stride(2),
        kv_nope.stride(0), kv_nope.stride(1), kv_nope.stride(2),
        k_rope.stride(0), k_rope.stride(1), k_rope.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        kv_len, scale,
        NH=nh,
        DKV=dkv,
        D_ROPE=d_rope,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return out


@triton.jit
def _mla_fused_attn_kernel_launcher(
    q_nope_latent_ptr, q_rope_ptr,
    kv_nope_ptr, k_rope_ptr,
    out_ptr,
    stride_qn_b, stride_qn_h, stride_qn_d,
    stride_qr_b, stride_qr_h, stride_qr_d,
    stride_kn_b, stride_kn_s, stride_kn_d,
    stride_kr_b, stride_kr_s, stride_kr_d,
    stride_o_b, stride_o_h, stride_o_d,
    kv_len,
    scale,
    NH: tl.constexpr,
    DKV: tl.constexpr,
    D_ROPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // NH
    h = pid - b * NH

    # Load q_nope_latent for this (b, h)
    q_nope_off = b * stride_qn_b + h * stride_qn_h
    q_nope = tl.load(
        q_nope_latent_ptr + q_nope_off + tl.arange(0, DKV) * stride_qn_d,
    ).to(tl.float32)

    # Load q_rope for this (b, h)
    q_rope_off = b * stride_qr_b + h * stride_qr_h
    q_rope = tl.load(
        q_rope_ptr + q_rope_off + tl.arange(0, D_ROPE) * stride_qr_d,
    ).to(tl.float32)

    # Online softmax state
    m_i = tl.full([1], -float("inf"), dtype=tl.float32)
    l_i = tl.full([1], 0.0, dtype=tl.float32)
    acc = tl.zeros([DKV], dtype=tl.float32)

    kv_nope_base = b * stride_kn_b
    k_rope_base = b * stride_kr_b

    for start_n in range(0, kv_len, BLOCK_N):
        n_off = start_n + tl.arange(0, BLOCK_N)
        mask_n = n_off < kv_len

        # Load kv_nope tile: [BLOCK_N, DKV]
        kv_nope_ptrs = (kv_nope_ptr + kv_nope_base
                        + n_off[:, None] * stride_kn_s
                        + tl.arange(0, DKV)[None, :] * stride_kn_d)
        kv_tile = tl.load(kv_nope_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # Load k_rope tile: [BLOCK_N, D_ROPE]
        k_rope_ptrs = (k_rope_ptr + k_rope_base
                       + n_off[:, None] * stride_kr_s
                       + tl.arange(0, D_ROPE)[None, :] * stride_kr_d)
        kr_tile = tl.load(k_rope_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # Compute scores
        scores_nope = tl.sum(q_nope[None, :] * kv_tile, axis=1)   # [BLOCK_N]
        scores_rope = tl.sum(q_rope[None, :] * kr_tile, axis=1)   # [BLOCK_N]
        scores = (scores_nope + scores_rope) * scale

        scores = tl.where(mask_n, scores, -float("inf"))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores))
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(scores - m_new)
        l_new = alpha * l_i + tl.sum(beta)

        acc = acc * alpha + tl.sum(beta[:, None] * kv_tile, axis=0)

        m_i = m_new
        l_i = l_new

    # Normalize
    acc = acc / l_i

    out_off = b * stride_o_b + h * stride_o_h
    tl.store(
        out_ptr + out_off + tl.arange(0, DKV) * stride_o_d,
        acc.to(tl.bfloat16),
    )


def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward step of the Multi-head Latent Attention (MLA) module.
    Uses a custom Triton fused attention kernel with online softmax (flash-attn style).
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
    q_lora = F.linear(x, wDQ)                      # [bs, 1, dq]
    kv_lora_input = F.linear(x, wDKV)              # [bs, 1, dkv+d_rope]

    kv_lora, kv_len = kv_cache(kv_lora_input)      # [bs, kv_len, dkv+d_rope]
    query_pos = kv_len - 1

    # Up-project queries
    q_up = F.linear(q_lora.squeeze(1), wUQ)        # [bs, nh*(d_nope+d_rope)]
    q_up = q_up.view(bs, nh, d_nope + d_rope)
    q_nope = q_up[..., :d_nope]                    # [bs, nh, d_nope]
    q_rope = q_up[..., d_nope:]                    # [bs, nh, d_rope]

    # Split KV latent
    kv_nope_input = kv_lora[..., :dkv].contiguous()   # [bs, kv_len, dkv]
    k_rope_input = kv_lora[..., dkv:]                  # [bs, kv_len, d_rope]

    # RoPE for keys and query
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)
    cos_q = cos_table[query_pos]
    sin_q = sin_table[query_pos]
    q_rope = (q_rope * cos_q + _rotate_half(q_rope) * sin_q).contiguous()

    cos_k = cos_table[:kv_len]
    sin_k = sin_table[:kv_len]
    k_rope = (k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k).contiguous()

    # Absorbed KV: q_nope_latent = q_nope @ wK -> [bs, nh, dkv]
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
    wK = wUKV_view[:, :d_nope, :].contiguous()         # [nh, d_nope, dkv]
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK).contiguous()  # [bs, nh, dkv]

    scale = 1.0 / math.sqrt(d_nope + d_rope)

    # Fused attention: online softmax over kv_len, accumulate V=kv_nope
    # Returns [bs, nh, dkv]
    M = _mla_fused_attn(
        q_nope_latent, q_rope,
        kv_nope_input, k_rope,
        scale=scale,
        BLOCK_N=64,
    )

    # Apply wV projection: wV [nh, dv, dkv], wV_T [nh, dkv, dv]
    wV = wUKV_view[:, d_nope:, :].contiguous()         # [nh, dv, dkv]
    wV_T = wV.permute(0, 2, 1).contiguous()            # [nh, dkv, dv]
    y_head = torch.einsum('bhd,hdk->bhk', M.to(torch.bfloat16), wV_T)  # [bs, nh, dv]

    y = y_head.reshape(bs, nh * dv)
    y = y.unsqueeze(1)
    output = F.linear(y, wO)

    return output, kv_cache.data
# EVOLVE-BLOCK-END
