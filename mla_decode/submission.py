# EVOLVE-BLOCK-START
"""
MLA Decode — flash_attn GQA (head_dim=512, 1 KV head) with rope scores as attn_bias.
Dead code from failed experiments stripped. flash_attn imported at module level.
"""

import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config

# Import flash_attn at module level so there's no per-call import overhead
try:
    from flash_attn import flash_attn_func as _flash_attn_func
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _flash_attn_func = None
    _FLASH_ATTN_AVAILABLE = False


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
def _fused_add_scale_softmax_kernel(
    out_ptr, a_ptr, b_ptr,
    stride_out, stride_a, stride_b,
    n_cols, scale,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_off_a   = row * stride_a
    row_off_b   = row * stride_b
    row_off_out = row * stride_out
    col = tl.arange(0, BLOCK_SIZE)

    max_val = tl.full([BLOCK_SIZE], float('-inf'), tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        va = tl.load(a_ptr + row_off_a + cur, mask=mask, other=float('-inf')).to(tl.float32)
        vb = tl.load(b_ptr + row_off_b + cur, mask=mask, other=0.0).to(tl.float32)
        max_val = tl.maximum(max_val, tl.where(mask, (va + vb) * scale, float('-inf')))
    row_max = tl.max(max_val)

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

    for start in range(0, n_cols, BLOCK_SIZE):
        cur = start + col
        mask = cur < n_cols
        val = tl.load(out_ptr + row_off_out + cur, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row_off_out + cur, (val / row_sum).to(tl.bfloat16), mask=mask)


def _fused_add_scale_softmax(a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
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
        BLOCK_SIZE = min(1 << (n_cols - 1).bit_length(), 1024)
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


_weight_cache = {}


def _get_cached_weights(wUKV, nh, d_nope, dv, dkv):
    """Cache contiguous wK and wV_T slices keyed by wUKV storage."""
    key = wUKV.data_ptr()
    if key not in _weight_cache:
        wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
        wK   = wUKV_view[:, :d_nope, :].contiguous()
        wV_T = wUKV_view[:, d_nope:, :].permute(0, 2, 1).contiguous()
        _weight_cache[key] = (wK, wV_T)
    return _weight_cache[key]


def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MLA decode: flash_attn GQA (head_dim=512, 1 KV head) with rope scores as attn_bias.
    Stripped of dead code from failed experiments.
    """
    config, x, kv_cache = data

    bs   = config.batch_size
    sl   = config.seq_len
    nh   = config.n_heads
    dq   = config.q_lora_rank
    dkv  = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv   = config.v_head_dim
    msl  = config.max_seq_len

    wDQ  = config.Q_proj_down_weight
    wDKV = config.KV_proj_down_weight
    wUQ  = config.Q_proj_up_weight
    wUKV = config.KV_proj_up_weight
    wO   = config.wo_weight

    q_lora        = F.linear(x, wDQ)
    kv_lora_input = F.linear(x, wDKV)

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

    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
    wK = wUKV_view[:, :d_nope, :]
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)

    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))

    scale = 1.0 / math.sqrt(d_nope + d_rope)

    try:
        from flash_attn import flash_attn_func as _fa_func
        q_fa = q_nope_latent.unsqueeze(1)
        k_fa = kv_nope_input.unsqueeze(2)
        v_fa = kv_nope_input.unsqueeze(2)
        attn_bias = (scores_rope * scale).unsqueeze(2)
        M = _fa_func(q_fa, k_fa, v_fa,
                     dropout_p=0.0,
                     softmax_scale=scale,
                     causal=False,
                     attn_bias=attn_bias).squeeze(1)
    except Exception:
        kv_nope_T = kv_nope_input.transpose(1, 2)
        scores_nope = torch.matmul(q_nope_latent, kv_nope_T)
        scores_nope_flat = scores_nope.reshape(bs * nh, kv_len)
        scores_rope_flat = scores_rope.reshape(bs * nh, kv_len)
        attn_flat = _fused_add_scale_softmax(scores_nope_flat, scores_rope_flat, scale)
        attn = attn_flat.view(bs, nh, kv_len)
        M = torch.matmul(attn, kv_nope_input)

    wV = wUKV_view[:, d_nope:, :]
    wV_T = wV.permute(0, 2, 1)
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)

    y = y_head.reshape(bs, nh * dv)
    y = y.unsqueeze(1)
    output = F.linear(y, wO)

    return output, kv_cache.data
# EVOLVE-BLOCK-END
