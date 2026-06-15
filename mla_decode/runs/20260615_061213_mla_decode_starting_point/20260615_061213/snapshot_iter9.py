# EVOLVE-BLOCK-START
"""
MLA Decode — weight absorption optimization (DeepSeek MLA standard approach).

Pre-fuse weight matrices offline to eliminate runtime einsum projections:
1. wUQ_absorbed = wUQ_nope_part fused with wK -> q_nope_latent via single F.linear
2. wVO_absorbed = wV fused with wO -> single F.linear from M to output
3. Keep contiguous KV copies (exp#8 showed non-contiguous is slower)
4. Keep Triton softmax (fp32 F.softmax was slower in exp#8)
"""

import math
from typing import Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import KVCache, Config


_rope_cache = {}
_weight_cache = {}


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


def _get_absorbed_weights(config):
    """
    Pre-compute absorbed weight matrices, cached by config id.

    Absorptions:
    - wUQ_nope_absorbed: [nh*dkv, dq] — replaces wUQ_nope + wK einsum
      q_nope_latent = F.linear(q_lora_sq, wUQ_nope_absorbed).view(bs, nh, dkv)

    - wUQ_rope: [nh*d_rope, dq] — rope part of wUQ (unchanged)
      q_rope_raw = F.linear(q_lora_sq, wUQ_rope).view(bs, nh, d_rope)

    - wVO_absorbed: [dim, nh*dkv] — replaces wV einsum + wO linear
      output = F.linear(M.reshape(bs, nh*dkv), wVO_absorbed) — but need reshape of M...
      Actually: output[bs,1,dim] = F.linear(M.reshape(bs,1,nh*dkv), wVO_absorbed)
    """
    key = id(config)
    if key in _weight_cache:
        return _weight_cache[key]

    nh = config.n_heads
    dq = config.q_lora_rank
    dkv = config.kv_lora_rank
    d_nope = config.qk_nope_head_dim
    d_rope = config.qk_rope_head_dim
    dv = config.v_head_dim
    dim = config.wo_weight.shape[0]

    wUQ = config.Q_proj_up_weight    # [nh*(d_nope+d_rope), dq]
    wUKV = config.KV_proj_up_weight  # [nh*(d_nope+dv)*dkv] -> view [nh, d_nope+dv, dkv]
    wO = config.wo_weight            # [dim, nh*dv]

    # Split wUQ into nope and rope parts
    # wUQ shape: [nh*(d_nope+d_rope), dq] — rows are interleaved per head
    # Reshape to [nh, d_nope+d_rope, dq]
    wUQ_per_head = wUQ.view(nh, d_nope + d_rope, dq)
    wUQ_nope = wUQ_per_head[:, :d_nope, :]   # [nh, d_nope, dq]
    wUQ_rope = wUQ_per_head[:, d_nope:, :]   # [nh, d_rope, dq]

    # wK from wUKV: [nh, d_nope, dkv]
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
    wK = wUKV_view[:, :d_nope, :].contiguous()  # [nh, d_nope, dkv]

    # Absorb: wUQ_nope_absorbed[h] = wK[h]^T @ wUQ_nope[h]
    # = [dkv, d_nope] @ [d_nope, dq] = [dkv, dq]
    # Full: [nh, dkv, dq] -> reshape to [nh*dkv, dq]
    wUQ_nope_absorbed = torch.bmm(
        wK.transpose(1, 2).float(),      # [nh, dkv, d_nope]
        wUQ_nope.float()                  # [nh, d_nope, dq]
    ).to(torch.bfloat16).reshape(nh * dkv, dq)  # [nh*dkv, dq]

    # wUQ_rope: reshape back to [nh*d_rope, dq]
    wUQ_rope_flat = wUQ_rope.reshape(nh * d_rope, dq)  # [nh*d_rope, dq]

    # Absorb wV into wO:
    # y_head [bs, nh, dv] = M [bs, nh, dkv] @ wV_T [nh, dkv, dv]
    # y [bs, 1, dim] = y_head.reshape(bs, nh*dv) @ wO.T
    # Fuse: output[bs, dim] = M.reshape(bs, nh*dkv) @ (wV_T_flat @ wO.T)
    # wV: [nh, dv, dkv], wV_T: [nh, dkv, dv]
    wV = wUKV_view[:, d_nope:, :].contiguous()   # [nh, dv, dkv]
    wV_T = wV.permute(0, 2, 1)                   # [nh, dkv, dv] — per-head [dkv, dv]
    # wV_T_flat: [nh*dkv, dv] (we need M [bs, nh, dkv] -> [bs, nh*dkv] @ [nh*dkv, dim])
    # wO: [dim, nh*dv] — F.linear weight: output = x @ wO.T
    # Need: [nh*dkv, dim] = wV_T_flat @ wO.T ... but wV_T is not [nh*dkv, dv] naively
    # wV_T[h]: [dkv, dv] but not the same h-ordering as wO
    # wO: [dim, nh*dv] — for head h, columns h*dv:(h+1)*dv
    # y_head[b,h,:] @ wO[row, h*dv:(h+1)*dv] contributes to output[b,row]
    # M[b,h,:] @ wV_T[h] -> y_head[b,h,:] -> y_head.reshape -> wO
    # Full: output[b,dim] = sum_h M[b,h,:] @ wV_T[h] @ wO[:,h*dv:(h+1)*dv].T
    # = M.reshape(bs, nh*dkv) @ block_diag(wV_T[0],...) @ wO.T
    # wVO per head: wV_T[h] @ wO[:, h*dv:(h+1)*dv].T = [dkv, dv] @ [dv, dim] = [dkv, dim]
    # Full: wVO_absorbed[h*dkv:(h+1)*dkv, :] = wV_T[h] @ wO[:, h*dv:(h+1)*dv].T
    wVO_absorbed = torch.zeros(nh * dkv, dim, dtype=torch.float32, device=wO.device)
    wO_f = wO.float()  # [dim, nh*dv]
    for h in range(nh):
        wVO_absorbed[h*dkv:(h+1)*dkv, :] = (
            wV_T[h].float() @          # [dkv, dv]
            wO_f[:, h*dv:(h+1)*dv].T  # [dv, dim]
        )
    wVO_absorbed = wVO_absorbed.to(torch.bfloat16).T.contiguous()  # [dim, nh*dkv]

    result = (wUQ_nope_absorbed, wUQ_rope_flat, wVO_absorbed)
    _weight_cache[key] = result
    return result


def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MLA decode with pre-absorbed weight matrices to eliminate runtime einsum.
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
    wUKV = config.KV_proj_up_weight

    # Get pre-absorbed weights (cached after first call)
    wUQ_nope_abs, wUQ_rope, wVO_abs = _get_absorbed_weights(config)

    # Down-projection GEMMs
    q_lora = F.linear(x, wDQ)                    # [bs, 1, dq]
    kv_lora_input = F.linear(x, wDKV)            # [bs, 1, dkv+d_rope]

    kv_lora, kv_len = kv_cache(kv_lora_input)    # [bs, kv_len, dkv+d_rope]
    query_pos = kv_len - 1

    q_lora_sq = q_lora.squeeze(1)                # [bs, dq]

    # q_nope_latent directly via absorbed weight: [bs, nh*dkv] -> [bs, nh, dkv]
    q_nope_latent = F.linear(q_lora_sq, wUQ_nope_abs).view(bs, nh, dkv)  # [bs, nh, dkv]

    # q_rope via rope part of wUQ: [bs, nh*d_rope] -> [bs, nh, d_rope]
    q_rope_raw = F.linear(q_lora_sq, wUQ_rope).view(bs, nh, d_rope)      # [bs, nh, d_rope]

    # KV cache slices (contiguous copies are faster per exp#8)
    kv_nope_input = kv_lora[..., :dkv].contiguous()   # [bs, kv_len, dkv]
    k_rope_input = kv_lora[..., dkv:].contiguous()    # [bs, kv_len, d_rope]

    # RoPE for query and keys
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)
    cos_q = cos_table[query_pos].view(d_rope).contiguous()
    sin_q = sin_table[query_pos].view(d_rope).contiguous()
    q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q  # [bs, nh, d_rope]

    cos_k = cos_table[:kv_len]
    sin_k = sin_table[:kv_len]
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k  # [bs, kv_len, d_rope]

    # Score matmuls (baseline structure: two separate, no per-head expansion)
    kv_nope_T = kv_nope_input.transpose(1, 2)    # [bs, dkv, kv_len]
    scores_nope = torch.matmul(q_nope_latent, kv_nope_T)              # [bs, nh, kv_len]
    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))      # [bs, nh, kv_len]

    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale

    # Triton softmax (bfloat16, proven baseline)
    scores_flat = scores.reshape(bs * nh, kv_len)
    attn_flat = _triton_softmax(scores_flat)
    attn = attn_flat.view(bs, nh, kv_len)

    # V aggregation: [bs, nh, kv_len] @ [bs, kv_len, dkv] -> [bs, nh, dkv]
    M = torch.matmul(attn, kv_nope_input)                              # [bs, nh, dkv]

    # Output via absorbed wVO: M [bs, nh*dkv] @ wVO_abs [nh*dkv, dim]
    output = F.linear(M.reshape(bs, 1, nh * dkv), wVO_abs)            # [bs, 1, dim]

    return output, kv_cache.data
# EVOLVE-BLOCK-END
