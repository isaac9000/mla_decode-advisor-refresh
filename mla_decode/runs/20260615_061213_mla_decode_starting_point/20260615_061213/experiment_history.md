# Experiment History

Tracks every kernel attempt, its code, hypothesis, and result.

---

## Experiment #1 — 2026-06-15 06:14:32 UTC ✅ KEEP

**Hypothesis:** Baseline 'starting_point' — initial benchmark

**Result:** 2740.78 μs

**Kernel code:**
```python
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


def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimised forward step of the Multi-head Latent Attention (MLA) module.
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

    q_lora = F.linear(x, wDQ)
    kv_lora_input = F.linear(x, wDKV)

    kv_lora, kv_len = kv_cache(kv_lora_input)
    query_pos = kv_len - 1

    q_up = F.linear(q_lora.squeeze(1), wUQ)
    q_up = q_up.view(bs, nh, d_nope + d_rope)
    q_nope = q_up[..., :d_nope]
    q_rope = q_up[..., d_nope:]

    kv_nope_input = kv_lora[..., :dkv]
    k_rope_input = kv_lora[..., dkv:]

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

    kv_nope_T = kv_nope_input.transpose(1, 2)
    scores_nope = torch.matmul(q_nope_latent, kv_nope_T)

    scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))

    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale

    scores_flat = scores.reshape(bs * nh, kv_len)
    attn_flat = _triton_softmax(scores_flat)
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

```

---

## Experiment #2 — 2026-06-15 06:24:17 UTC ❌ DISCARD

**Hypothesis:** Advisor proposed replacing score/softmax/V-matmul chain with FlashAttention via torch.nn.functional.scaled_dot_product_attention. Implemented composite Q=[q_nope_latent|q_rope] and K=[kv_nope|k_rope] approach, then expanded K and V across n_heads=128 for SDPA. Result was much slower (5022 µs vs 2740 µs baseline) because expanding K/V to [bs, nh, kv_len, dkv+d_rope] = [128, 128, kv_len, 576] creates massive tensors that overwhelm memory bandwidth, negating any SDPA benefit.

**Result:** 5022.43 μs

**Kernel code:**
```python
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

```

---

## Experiment #3 — 2026-06-15 06:28:12 UTC 💥 CRASH

**Hypothesis:** Advisor proposed using SDPA in MQA/GQA mode: Q=[bs,128,1,576], K/V=[bs,1,kv_len,576] with no expand, relying on PyTorch SDPA's native GQA support. Tests passed (short sequences), but benchmarking with kv_len=4096/6144 caused OOM (tried to allocate 144 GiB). PyTorch SDPA's GQA path with head_dim=576 apparently materializes a huge intermediate tensor. The test sequences were short enough to pass but benchmark sequences were too long.

**Result:** CRASH

**Error:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 144.04 GiB. GPU 0 has a total capacity of 139.80 GiB. PyTorch SDPA GQA mode with Q=[bs,128,1,576] K/V=[bs,1,kv_len,576] OOMs at kv_len=4096/6144.
```

**Kernel code:**
```python
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

    # MQA/GQA mode: 1 KV group, 128 Q heads — no per-head expansion of K/V
    # SDPA supports GQA natively: Q=[bs, nh, sq, hd], K/V=[bs, 1, kv_len, hd]

    # Q: [bs, nh, 1, dkv+d_rope]
    Q_comp = torch.cat([q_nope_latent, q_rope], dim=-1).unsqueeze(2).contiguous()

    # K: [bs, 1, kv_len, dkv+d_rope] — single KV group, no expand
    K_comp = torch.cat([kv_nope_input, k_rope], dim=-1).unsqueeze(1).contiguous()

    # V: [bs, 1, kv_len, dkv] — single KV group, no expand
    V_comp = kv_nope_input.unsqueeze(1).contiguous()

    scale = 1.0 / math.sqrt(d_nope + d_rope)

    # FlashAttention via SDPA in MQA mode (1 KV head, 128 Q heads)
    # PyTorch SDPA handles GQA when n_heads_q is a multiple of n_heads_kv
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

```

---

## Experiment #4 — 2026-06-15 06:33:43 UTC ❌ DISCARD

**Hypothesis:** Advisor proposed a custom Triton fused attention kernel with online softmax (flash-attn style), grid (bs*nh,), streaming over kv_len in tiles. Implemented as proposed: one program per (batch,head), online softmax, accumulate V=kv_nope. Result was 95770 µs — 35x worse than baseline. Root cause: the [BLOCK_N=64, DKV=512] tile operations dominate, and the serial loop over kv_len with large DKV inner dimension creates massive register pressure and poor SIMT utilization. The approach is fundamentally memory-bound in the wrong way — loading 512D vectors per tile is too much sequential work per thread.

**Result:** 95770.18 μs

**Kernel code:**
```python
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

```

---

## Experiment #5 — 2026-06-15 06:39:21 UTC 💥 CRASH

**Hypothesis:** Advisor proposed replacing einsum with explicit bmm and Triton softmax with F.softmax for better cuBLAS efficiency. Implemented as proposed but OOM at kv_nope_input.unsqueeze(1).expand(-1,nh,-1,-1).reshape(bs*nh, kv_len, dkv) — 64 GiB allocation. The reshape() after expand() forces materialization of [16384, kv_len, 512] tensor. The fundamental issue is that any approach requiring per-head K/V materialization will OOM at benchmark sequence lengths. The score computation must stay in [bs, nh, kv_len] form using broadcasting, not per-head-expanded K/V.

**Result:** CRASH

**Error:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.02 GiB. kv_nope_input.expand(-1,nh,-1,-1).reshape(bs*nh, kv_len, dkv) materializes [16384, kv_len, 512] tensor.
```

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
MLA Decode — optimized PyTorch-native ops: explicit bmm, F.softmax, contiguous layouts.

Key changes vs baseline:
- Replace einsum('bhd,hdk->bhk') with explicit reshape+bmm for q_nope_latent and wV
- Replace custom 3-pass Triton softmax with F.softmax (cuDNN fused)
- Ensure contiguous tensors before all matmul/bmm calls
- Remove Triton RoPE kernel, use pure PyTorch for query RoPE too
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
    Optimized MLA decode: explicit bmm instead of einsum, F.softmax instead of Triton softmax.
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

    # RoPE for query and keys (pure PyTorch)
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)
    cos_q = cos_table[query_pos]   # [d_rope]
    sin_q = sin_table[query_pos]   # [d_rope]
    q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q  # [bs, nh, d_rope]

    cos_k = cos_table[:kv_len]    # [kv_len, d_rope]
    sin_k = sin_table[:kv_len]    # [kv_len, d_rope]
    k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k  # [bs, kv_len, d_rope]

    # Absorbed KV: q_nope_latent = q_nope @ wK -> [bs, nh, dkv]
    # Use bmm: reshape q_nope to [bs*nh, 1, d_nope], wK to [bs*nh, d_nope, dkv]
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
    wK = wUKV_view[:, :d_nope, :].contiguous()       # [nh, d_nope, dkv]
    q_nope_flat = q_nope.reshape(bs * nh, 1, d_nope)
    wK_exp = wK.unsqueeze(0).expand(bs, -1, -1, -1).reshape(bs * nh, d_nope, dkv)
    q_nope_latent = torch.bmm(q_nope_flat, wK_exp).squeeze(1)  # [bs*nh, dkv]

    # scores_nope: [bs*nh, dkv] @ [bs*nh, dkv, kv_len] -> [bs*nh, kv_len]
    kv_nope_T = kv_nope_input.transpose(1, 2)        # [bs, dkv, kv_len]
    kv_nope_T_exp = kv_nope_T.unsqueeze(1).expand(-1, nh, -1, -1).reshape(bs * nh, dkv, kv_len)
    scores_nope = torch.bmm(q_nope_latent.unsqueeze(1), kv_nope_T_exp).squeeze(1)  # [bs*nh, kv_len]

    # scores_rope: [bs*nh, d_rope] @ [bs*nh, d_rope, kv_len] -> [bs*nh, kv_len]
    k_rope_T = k_rope.transpose(1, 2)                # [bs, d_rope, kv_len]
    q_rope_flat = q_rope.reshape(bs * nh, 1, d_rope)
    k_rope_T_exp = k_rope_T.unsqueeze(1).expand(-1, nh, -1, -1).reshape(bs * nh, d_rope, kv_len)
    scores_rope = torch.bmm(q_rope_flat, k_rope_T_exp).squeeze(1)  # [bs*nh, kv_len]

    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale     # [bs*nh, kv_len]

    # F.softmax with cuDNN fused kernel (faster than 3-pass Triton softmax)
    attn = F.softmax(scores.float(), dim=-1).to(torch.bfloat16)  # [bs*nh, kv_len]

    # Weighted V sum: attn [bs*nh, 1, kv_len] @ kv_nope [bs*nh, kv_len, dkv] -> [bs*nh, dkv]
    kv_nope_exp = kv_nope_input.unsqueeze(1).expand(-1, nh, -1, -1).reshape(bs * nh, kv_len, dkv)
    M = torch.bmm(attn.unsqueeze(1), kv_nope_exp).squeeze(1)    # [bs*nh, dkv]

    # wV projection: wV [nh, dv, dkv] -> wV_T [nh, dkv, dv]
    wV = wUKV_view[:, d_nope:, :].contiguous()       # [nh, dv, dkv]
    wV_T = wV.permute(0, 2, 1).contiguous()          # [nh, dkv, dv]
    M_flat = M.unsqueeze(1)                           # [bs*nh, 1, dkv]
    wV_T_exp = wV_T.unsqueeze(0).expand(bs, -1, -1, -1).reshape(bs * nh, dkv, dv)
    y_head = torch.bmm(M_flat, wV_T_exp).squeeze(1)  # [bs*nh, dv]

    y = y_head.reshape(bs, nh * dv).unsqueeze(1)
    output = F.linear(y, wO)

    return output, kv_cache.data
# EVOLVE-BLOCK-END

```

---

## Experiment #6 — 2026-06-15 06:43:55 UTC ❌ DISCARD

**Hypothesis:** Advisor proposed xformers memory_efficient_attention with Q_rope[bs,1,nh,64], K_rope[bs,kv_len,1,64], V[bs,kv_len,1,512], and scores_nope[bs,nh,1,kv_len] as attn_bias. xformers was available on Modal. Result: 4741 µs, slower than baseline 2740 µs. Likely causes: (1) scores_nope full matmul still happens before MEA, (2) attn_bias with mismatched Q_dim=64 vs V_dim=512 likely forces xformers onto a non-FlashAttention fallback path, (3) overhead of tensor reshaping for xformers format. The fallback path (F.softmax) was not exercised since xformers was found.

**Result:** 4741.22 μs

**Kernel code:**
```python
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

```

---

## Experiment #7 — 2026-06-15 06:50:20 UTC ❌ DISCARD

**Hypothesis:** Advisor proposed: (1) single fused score matmul via Q_combined=[q_nope_latent|q_rope] cat and K_combined=[kv_nope|k_rope] cat, (2) F.softmax replacing Triton softmax, (3) matmul replacing einsum. Result: 10351 µs, 3.8x worse than baseline. Root cause: torch.cat to build K_combined materializes a full [bs, kv_len, 576] copy of the KV cache data (expensive alloc+copy), and the 576-wide matmul is less efficient than two separate narrower ones. The unsqueeze/squeeze pattern for the weight matmuls also adds overhead. The single-matmul 'optimization' is actually slower due to the required cat allocation.

**Result:** 10351.26 μs

**Kernel code:**
```python
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

```

---

## Experiment #8 — 2026-06-15 06:55:22 UTC ❌ DISCARD

**Hypothesis:** Advisor proposed eliminating .contiguous() copies for kv_nope/k_rope slices (using non-contiguous views), replacing 3-pass Triton softmax with F.softmax(fp32), keeping baseline einsum structure. Result: 3809 µs, slower than baseline 2740 µs. Non-contiguous KV slices likely cause PyTorch matmul to do implicit internal copies (cuBLAS requires contiguous), negating the savings. The F.softmax fp32 cast also adds overhead vs bfloat16 Triton softmax. The baseline's explicit .contiguous() copies are actually faster because they happen once upfront rather than being repeated inside each matmul call.

**Result:** 3809.76 μs

**Kernel code:**
```python
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

```

---

## Experiment #9 — 2026-06-15 07:00:13 UTC ❌ DISCARD

**Hypothesis:** Advisor proposed pre-absorbing wUQ×wK into wUQ_nope_absorbed [nh*dkv, dq]=[65536,1536] and wV×wO into wVO_absorbed [dim, nh*dkv]=[7168,65536]. These eliminate the runtime einsum calls but create much larger weight matrices. Result: 4081 µs vs baseline 2740 µs. The larger GEMM F.linear(q_lora, wUQ_nope_abs) [bs,1536]@[1536,65536] and F.linear(M, wVO_abs) [bs,65536]@[65536,7168] are slower than the original smaller GEMMs plus the einsum contractions. The absorption trades a cheap einsum for expensive large GEMMs.

**Result:** 4081.16 μs

**Kernel code:**
```python
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

```

---

## Experiment #10 — 2026-06-15 07:06:57 UTC ❌ DISCARD

**Hypothesis:** Advisor proposed fused RoPE+rope_scores Triton kernel (grid bs*nh, serial loop over kv_len) to eliminate [bs,kv_len,64] k_rope materialization. Implemented with constexpr NH fix from kernel syntax error. Result: 6922 µs, 2.5x slower than baseline. Root cause: loads k_raw twice per tile (once for main, once for rotate_half shifted indices), doubling HBM bandwidth vs baseline. Also same serial-loop issue as exp#4 — bs*nh=16384 programs each making ~64-96 serial HBM fetches. The fused approach is strictly worse than the two-step baseline (materialize k_rope, then matmul).

**Result:** 6922.98 μs

**Kernel code:**
```python
# EVOLVE-BLOCK-START
"""
MLA Decode — fused RoPE+rope_scores Triton kernel eliminates k_rope materialization.

Key change vs baseline:
- Replace: k_rope = rotate(k_rope_input) then scores_rope = q_rope @ k_rope.T
- With: single Triton kernel that computes scores_rope[b,h,t] = q_rope[b,h,:] · rot(k_raw[b,t,:], t)
  without ever writing the full [bs, kv_len, 64] rotated k_rope tensor to HBM
- Saves ~100MB allocation + write bandwidth; 64-dim rope dot fits in registers
- Everything else identical to baseline (contiguous kv_nope, Triton softmax, einsum projections)
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
def _rope_scores_kernel(
    # Inputs
    q_rope_ptr,     # [bs, nh, d_rope]  — rotated query rope
    k_raw_ptr,      # [bs, kv_len, d_rope]  — raw (un-rotated) key rope from KV cache
    cos_ptr,        # [kv_len, d_rope]
    sin_ptr,        # [kv_len, d_rope]
    # Output
    scores_ptr,     # [bs, nh, kv_len]
    # Strides
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_b, stride_k_t, stride_k_d,
    stride_cs_t, stride_cs_d,           # cos/sin share same strides
    stride_s_b, stride_s_h, stride_s_t,
    # Dims
    kv_len,
    NH: tl.constexpr,
    D_ROPE: tl.constexpr,   # 64
    BLOCK_T: tl.constexpr,  # tile over kv_len
):
    """
    One program = one (batch, head) pair.
    For each t in kv_len: apply RoPE to k_raw[b,t,:], dot with q_rope[b,h,:], write score.
    Avoids materializing rotated k_rope entirely.
    """
    pid = tl.program_id(0)
    b = pid // NH
    h = pid - b * NH

    # Load q_rope[b, h, :]: shape [D_ROPE]
    q_off = b * stride_q_b + h * stride_q_h
    q = tl.load(q_rope_ptr + q_off + tl.arange(0, D_ROPE) * stride_q_d).to(tl.float32)

    half = D_ROPE // 2
    d_idx = tl.arange(0, D_ROPE)

    # Loop over kv_len in tiles
    for start_t in range(0, kv_len, BLOCK_T):
        t_idx = start_t + tl.arange(0, BLOCK_T)
        mask_t = t_idx < kv_len

        # Load k_raw tile: [BLOCK_T, D_ROPE]
        k_ptrs = (k_raw_ptr + b * stride_k_b
                  + t_idx[:, None] * stride_k_t
                  + d_idx[None, :] * stride_k_d)
        k_raw_tile = tl.load(k_ptrs, mask=mask_t[:, None], other=0.0).to(tl.float32)

        # Load cos/sin tile: [BLOCK_T, D_ROPE]
        cos_tile = tl.load(
            cos_ptr + t_idx[:, None] * stride_cs_t + d_idx[None, :] * stride_cs_d,
            mask=mask_t[:, None], other=0.0).to(tl.float32)
        sin_tile = tl.load(
            sin_ptr + t_idx[:, None] * stride_cs_t + d_idx[None, :] * stride_cs_d,
            mask=mask_t[:, None], other=0.0).to(tl.float32)

        # Apply rotate_half: load first and second halves separately
        # For d < half: rot[t,d] = -k_raw[t, d+half]
        # For d >= half: rot[t,d] = k_raw[t, d-half]
        # Load k_raw again with shifted indices
        d_shifted = tl.where(d_idx < half, d_idx + half, d_idx - half)
        k_shift_ptrs = (k_raw_ptr + b * stride_k_b
                        + t_idx[:, None] * stride_k_t
                        + d_shifted[None, :] * stride_k_d)
        k_shifted = tl.load(k_shift_ptrs, mask=mask_t[:, None], other=0.0).to(tl.float32)
        k_neg_half = tl.where(d_idx[None, :] < half, -k_shifted, k_shifted)

        k_rot = k_raw_tile * cos_tile + k_neg_half * sin_tile  # [BLOCK_T, D_ROPE]

        # Dot product: scores[b, h, t] = sum_d q[d] * k_rot[t, d]
        scores_tile = tl.sum(q[None, :] * k_rot, axis=1)  # [BLOCK_T]

        # Store
        s_ptrs = scores_ptr + b * stride_s_b + h * stride_s_h + t_idx * stride_s_t
        tl.store(s_ptrs, scores_tile.to(tl.bfloat16), mask=mask_t)


def _rope_scores_triton(
    q_rope: torch.Tensor,    # [bs, nh, d_rope]  — already rotated
    k_raw: torch.Tensor,     # [bs, kv_len, d_rope]  — raw from KV cache
    cos_table: torch.Tensor, # [kv_len, d_rope]
    sin_table: torch.Tensor, # [kv_len, d_rope]
) -> torch.Tensor:
    bs, nh, d_rope = q_rope.shape
    kv_len = k_raw.shape[1]

    scores = torch.empty((bs, nh, kv_len), dtype=torch.bfloat16, device=q_rope.device)

    q_rope_c = q_rope.contiguous()
    k_raw_c = k_raw.contiguous()
    cos_c = cos_table.contiguous()
    sin_c = sin_table.contiguous()

    grid = (bs * nh,)
    BLOCK_T = 64
    _rope_scores_kernel[grid](
        q_rope_c, k_raw_c, cos_c, sin_c, scores,
        q_rope_c.stride(0), q_rope_c.stride(1), q_rope_c.stride(2),
        k_raw_c.stride(0), k_raw_c.stride(1), k_raw_c.stride(2),
        cos_c.stride(0), cos_c.stride(1),
        scores.stride(0), scores.stride(1), scores.stride(2),
        kv_len,
        NH=nh,
        D_ROPE=d_rope,
        BLOCK_T=BLOCK_T,
        num_warps=4,
        num_stages=2,
    )
    return scores


def custom_kernel(data: Tuple[Config, torch.Tensor, KVCache]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MLA decode: baseline structure + fused RoPE+dot Triton kernel for rope scores.
    Eliminates k_rope [bs, kv_len, 64] materialization.
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
    q_rope_raw = q_up[..., d_nope:]             # [bs, nh, d_rope]

    # KV slices — contiguous (faster per exp#8)
    kv_nope_input = kv_lora[..., :dkv].contiguous()  # [bs, kv_len, dkv]
    k_rope_raw = kv_lora[..., dkv:].contiguous()     # [bs, kv_len, d_rope]  (un-rotated)

    # RoPE tables
    cos_table, sin_table = _get_rope_tables(d_rope, msl, x.device)

    # Apply RoPE to query only
    cos_q = cos_table[query_pos].view(d_rope).contiguous()
    sin_q = sin_table[query_pos].view(d_rope).contiguous()
    q_rope = q_rope_raw * cos_q + _rotate_half(q_rope_raw) * sin_q  # [bs, nh, d_rope]

    # Absorbed KV: q_nope_latent = einsum('bhd,hdk->bhk', q_nope, wK) -> [bs, nh, dkv]
    wUKV_view = wUKV.view(nh, d_nope + dv, dkv)
    wK = wUKV_view[:, :d_nope, :]
    q_nope_latent = torch.einsum('bhd,hdk->bhk', q_nope, wK)

    # Score nope via standard matmul (baseline)
    kv_nope_T = kv_nope_input.transpose(1, 2)                        # [bs, dkv, kv_len]
    scores_nope = torch.matmul(q_nope_latent, kv_nope_T)             # [bs, nh, kv_len]

    # Score rope via fused RoPE+dot kernel — no k_rope materialization
    cos_k = cos_table[:kv_len].contiguous()   # [kv_len, d_rope]
    sin_k = sin_table[:kv_len].contiguous()   # [kv_len, d_rope]
    scores_rope = _rope_scores_triton(q_rope, k_rope_raw, cos_k, sin_k)  # [bs, nh, kv_len]

    scale = 1.0 / math.sqrt(d_nope + d_rope)
    scores = (scores_nope + scores_rope) * scale

    # Triton softmax (baseline)
    scores_flat = scores.reshape(bs * nh, kv_len)
    attn_flat = _triton_softmax(scores_flat)
    attn = attn_flat.view(bs, nh, kv_len)

    # V aggregation
    M = torch.matmul(attn, kv_nope_input)                             # [bs, nh, dkv]

    # Output projection (baseline)
    wV = wUKV_view[:, d_nope:, :]
    wV_T = wV.permute(0, 2, 1)
    y_head = torch.einsum('bhd,hdk->bhk', M, wV_T)                   # [bs, nh, dv]

    y = y_head.reshape(bs, nh * dv).unsqueeze(1)
    output = F.linear(y, wO)

    return output, kv_cache.data
# EVOLVE-BLOCK-END

```

---

## Experiment #11 — 2026-06-15 07:11:48 UTC ❌ DISCARD

**Hypothesis:** Advisor proposed: (1) replace torch.cat-based _rotate_half with torch.roll+sign-flip to avoid allocation, (2) BF16 torch.softmax instead of Triton softmax (no fp32 cast, no reshape). Result: 4887 µs vs baseline 2740 µs. torch.roll+in-place-negate is slower than torch.cat for _rotate_half (roll is not elementwise-fused). BF16 softmax without fp32 cast is numerically valid but PyTorch's softmax dispatch may be slower in BF16 than the Triton 3-pass kernel at these sequence lengths. Both changes hurt vs baseline.

**Result:** 4887.68 μs

**Kernel code:**
```python
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

```

