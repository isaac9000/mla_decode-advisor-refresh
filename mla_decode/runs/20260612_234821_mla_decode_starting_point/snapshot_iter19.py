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
def _mla_flash_kernel(
    # q_nope_latent [bs, nh, DKV], q_rope [bs, nh, DROPE]
    qnl_ptr, stride_qnl_b, stride_qnl_h, stride_qnl_d,
    qr_ptr,  stride_qr_b,  stride_qr_h,  stride_qr_d,
    # kv_nope [bs, S, DKV], k_rope [bs, S, DROPE]
    kvn_ptr, stride_kvn_b, stride_kvn_s, stride_kvn_d,
    kr_ptr,  stride_kr_b,  stride_kr_s,  stride_kr_d,
    # output [bs, nh, DKV]
    out_ptr, stride_ob, stride_oh, stride_od,
    kv_len, scale, NH,
    DKV: tl.constexpr,   # 512
    DROPE: tl.constexpr, # 64
    BLOCK_KV: tl.constexpr,  # tile over kv sequence
    BLOCK_D: tl.constexpr,   # 128 — DKV/4, for accumulator chunking
):
    """
    One CTA per (batch, head). Online softmax flash-attention.
    Accumulator split into 4 BLOCK_D=128 chunks to manage register pressure.
    Boundary fix: use full kv_len range with mask (avoids non-constant range bounds).
    """
    pid = tl.program_id(0)
    b = pid // NH
    h = pid % NH

    qnl_base = qnl_ptr + b * stride_qnl_b + h * stride_qnl_h
    qr_base  = qr_ptr  + b * stride_qr_b  + h * stride_qr_h
    kvn_base = kvn_ptr + b * stride_kvn_b
    kr_base  = kr_ptr  + b * stride_kr_b

    d_offs  = tl.arange(0, BLOCK_D)
    r_offs  = tl.arange(0, DROPE)
    kv_offs = tl.arange(0, BLOCK_KV)

    # Load q vectors
    qn0 = tl.load(qnl_base + (0*BLOCK_D + d_offs) * stride_qnl_d).to(tl.float32)
    qn1 = tl.load(qnl_base + (1*BLOCK_D + d_offs) * stride_qnl_d).to(tl.float32)
    qn2 = tl.load(qnl_base + (2*BLOCK_D + d_offs) * stride_qnl_d).to(tl.float32)
    qn3 = tl.load(qnl_base + (3*BLOCK_D + d_offs) * stride_qnl_d).to(tl.float32)
    qr  = tl.load(qr_base + r_offs * stride_qr_d).to(tl.float32)

    # Online softmax state + 4 output accumulators
    m_i  = tl.full([1], float('-inf'), tl.float32)
    l_i  = tl.full([1], 0.0, tl.float32)
    acc0 = tl.zeros([BLOCK_D], tl.float32)
    acc1 = tl.zeros([BLOCK_D], tl.float32)
    acc2 = tl.zeros([BLOCK_D], tl.float32)
    acc3 = tl.zeros([BLOCK_D], tl.float32)

    for start in range(0, kv_len, BLOCK_KV):
        s_offs = start + kv_offs
        mask   = s_offs < kv_len

        kr_tile = tl.load(kr_base  + s_offs[:, None] * stride_kr_s  + r_offs[None, :] * stride_kr_d,  mask=mask[:, None], other=0.0).to(tl.float32)
        kv0     = tl.load(kvn_base + s_offs[:, None] * stride_kvn_s + (0*BLOCK_D + d_offs[None, :]) * stride_kvn_d, mask=mask[:, None], other=0.0).to(tl.float32)
        kv1     = tl.load(kvn_base + s_offs[:, None] * stride_kvn_s + (1*BLOCK_D + d_offs[None, :]) * stride_kvn_d, mask=mask[:, None], other=0.0).to(tl.float32)
        kv2     = tl.load(kvn_base + s_offs[:, None] * stride_kvn_s + (2*BLOCK_D + d_offs[None, :]) * stride_kvn_d, mask=mask[:, None], other=0.0).to(tl.float32)
        kv3     = tl.load(kvn_base + s_offs[:, None] * stride_kvn_s + (3*BLOCK_D + d_offs[None, :]) * stride_kvn_d, mask=mask[:, None], other=0.0).to(tl.float32)

        scores = (tl.sum(kv0 * qn0[None, :], 1) + tl.sum(kv1 * qn1[None, :], 1) +
                  tl.sum(kv2 * qn2[None, :], 1) + tl.sum(kv3 * qn3[None, :], 1) +
                  tl.sum(kr_tile * qr[None, :], 1)) * scale
        scores = tl.where(mask, scores, float('-inf'))

        m_new = tl.maximum(m_i, tl.max(scores, 0))
        alpha = tl.exp(m_i - m_new)
        p     = tl.where(mask, tl.exp(scores - m_new), 0.0)
        l_new = alpha * l_i + tl.sum(p, 0)

        acc0 = alpha * acc0 + tl.sum(p[:, None] * kv0, 0)
        acc1 = alpha * acc1 + tl.sum(p[:, None] * kv1, 0)
        acc2 = alpha * acc2 + tl.sum(p[:, None] * kv2, 0)
        acc3 = alpha * acc3 + tl.sum(p[:, None] * kv3, 0)
        m_i  = m_new
        l_i  = l_new

    acc0 = acc0 / l_i;  acc1 = acc1 / l_i
    acc2 = acc2 / l_i;  acc3 = acc3 / l_i

    out_base = out_ptr + b * stride_ob + h * stride_oh
    tl.store(out_base + (0*BLOCK_D + d_offs) * stride_od, acc0.to(tl.bfloat16))
    tl.store(out_base + (1*BLOCK_D + d_offs) * stride_od, acc1.to(tl.bfloat16))
    tl.store(out_base + (2*BLOCK_D + d_offs) * stride_od, acc2.to(tl.bfloat16))
    tl.store(out_base + (3*BLOCK_D + d_offs) * stride_od, acc3.to(tl.bfloat16))


def _mla_triton_attn(q_nope_latent, q_rope, kv_nope, k_rope, scale, nh):
    bs, _, dkv = q_nope_latent.shape
    kv_len     = kv_nope.shape[1]
    d_rope     = q_rope.shape[2]
    out        = torch.empty(bs, nh, dkv, dtype=torch.bfloat16, device=kv_nope.device)
    _mla_flash_kernel[(bs * nh,)](
        q_nope_latent, q_nope_latent.stride(0), q_nope_latent.stride(1), q_nope_latent.stride(2),
        q_rope,        q_rope.stride(0),        q_rope.stride(1),        q_rope.stride(2),
        kv_nope,       kv_nope.stride(0),       kv_nope.stride(1),       kv_nope.stride(2),
        k_rope,        k_rope.stride(0),        k_rope.stride(1),        k_rope.stride(2),
        out,           out.stride(0),           out.stride(1),           out.stride(2),
        kv_len, scale, NH=nh,
        DKV=dkv, DROPE=d_rope, BLOCK_KV=32, BLOCK_D=128,
        num_warps=4,
    )
    return out


def _attention_inner(
    q_nope, q_rope, kv_nope_input, k_rope_input,
    cos_q, sin_q, cos_k, sin_k,
    wK, wV_T, wO,
    bs, nh, dkv, d_nope, d_rope, dv, scale,
):
    """Fallback PyTorch attention (used if Triton kernel disabled)."""
    q_rope = q_rope * cos_q + torch.cat((-q_rope[..., d_rope//2:], q_rope[..., :d_rope//2]), dim=-1) * sin_q
    k_rope_half = torch.cat((-k_rope_input[..., d_rope//2:], k_rope_input[..., :d_rope//2]), dim=-1)
    k_rope = k_rope_input * cos_k + k_rope_half * sin_k
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)
    kv_nope_T = kv_nope_input.transpose(1, 2)
    scores_nope = torch.matmul(q_nope_latent, kv_nope_T)
    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))
    scores = (scores_nope + scores_rope) * scale
    attn = torch.softmax(scores, dim=-1)
    M = torch.matmul(attn, kv_nope_input)
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)
    y = y_head.reshape(bs, nh * dv).unsqueeze(1)
    return F.linear(y, wO)


_compiled_attention = torch.compile(
    _attention_inner, mode='max-autotune-no-cudagraphs', dynamic=True
)


def _rotate_half_torch(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised MLA decode using Triton flash-attention kernel.
    Fixes from exp #9: uses full kv_len range with mask (no non-constant bounds),
    BLOCK_KV=32 to reduce register pressure, correct scalar lse handling.
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
    wUKV = config.KV_proj_up_weight
    wO   = config.wo_weight

    q_lora = F.linear(x, wDQ)
    kv_lora_input = F.linear(x, wDKV)

    kv_lora, kv_len = kv_cache(kv_lora_input)
    query_pos = kv_len - 1

    q_up = F.linear(q_lora.squeeze(1), wUQ)
    q_up = q_up.view(bs, nh, d_nope + d_rope)
    q_nope = q_up[..., :d_nope]
    q_rope_raw = q_up[..., d_nope:]

    kv_nope_input = kv_lora[..., :dkv]
    k_rope_input  = kv_lora[..., dkv:]

    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)
    cos_q = cos_table[query_pos]
    sin_q = sin_table[query_pos]
    cos_k = cos_table[:kv_len]
    sin_k = sin_table[:kv_len]

    # Apply RoPE to queries and keys (PyTorch, before Triton kernel)
    q_rope = (q_rope_raw * cos_q + _rotate_half_torch(q_rope_raw) * sin_q).contiguous()
    k_rope = (k_rope_input * cos_k + _rotate_half_torch(k_rope_input) * sin_k).contiguous()

    # Absorb wK into q_nope (PyTorch einsum, before Triton kernel)
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
    wK   = wUKV_view[:, :d_nope, :]
    wV   = wUKV_view[:, d_nope:, :]
    wV_T = wV.permute(0, 2, 1)

    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK).contiguous()  # [bs, nh, dkv]
    kv_nope_c = kv_nope_input.contiguous()  # [bs, kv_len, dkv]

    scale = 1.0 / math.sqrt(d_nope + d_rope)

    # Triton flash-attention: output M [bs, nh, dkv]
    M = _mla_triton_attn(q_nope_latent, q_rope, kv_nope_c, k_rope, scale, nh)

    # Project through wV then wO (PyTorch)
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)
    y = y_head.reshape(bs, nh * dv).unsqueeze(1)
    output = F.linear(y, wO)

    return output, kv_cache.data
# EVOLVE-BLOCK-END
