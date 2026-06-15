# Advisor Proposals

---

## Iteration 1

## STATE
Only one data point exists: the baseline at 2740.78 µs, ~13× above the roofline SOL (~210–281 µs). This is very early — no optimized attempts have been tried yet. The baseline uses naïve PyTorch ops (F.linear, matmul, einsum) plus simple Triton softmax/RoPE kernels, with no attention to memory layout, kernel fusion, or flash-attention-style tiling. The entire budget is open.

## RATIONALE
At 2740 µs vs. a ~245 µs geometric-mean SOL, the dominant bottleneck is almost certainly the attention computation: materializing `[128, 128, 8192]` score tensors in full, plus the scattered einsum/matmul chain for the absorbed KV projections. The immediate high-value move is to replace the attention phase with a fused Flash-Attention-style kernel (via `flash_attn` or `xformers`), while also ensuring the large GEMMs (wDQ@x, wDKV@x, wUQ, wO) are done in contiguous BF16 layout. The Triton softmax is marginally helpful but the multi-pass loop over an 8192-length sequence row is a clear memory-bandwidth bottleneck that flash attention resolves by fusing QK^T, softmax, and V-accumulation in a single tiled pass that stays in SRAM.

## PROPOSAL

**Fuse the attention core with FlashAttention + clean up the GEMM chain.**

1. **Replace the score/softmax/V-matmul sequence** with a `flash_attn_func` (or `flash_attn_with_kvcache`) call. This eliminates materializing the full `[bs, n_heads, seq_len]` score tensor and replaces three separate memory round-trips with a single tiled pass. The key challenge is constructing the right Q, K, V tensors in the layout flash_attn expects: `[bs, seq_len, n_heads, head_dim]`.

2. **For the absorbed-KV trick:** the nope scores come from `q_nope_latent @ kv_nope^T` where `q_nope_latent = q_nope @ wK`. Rather than splitting into two matmuls and summing, construct a composite Q of dimension `d_nope + d_rope` (the absorbed nope component concatenated with the rope component) and a composite K that is `[kv_nope | k_rope]` (the kv latent and rope key side by side), so that flash_attn computes the full dot product in one fused kernel with the correct combined head_dim.

3. **For the V side:** after flash_attn returns the weighted sum over kv latents, apply the `wV` projection (`[nh, dkv, dv]`) once as a batched GEMM rather than via einsum, since einsum emits less-optimal contraction order.

4. **Ensure all weight matrices and activations are contiguous** before large GEMM calls — non-contiguous tensors force implicit copies inside cuBLAS.

The goal of this iteration is to measure the ceiling of the flash-attention path. Don't over-tune block sizes yet — just get a correct fused-attention implementation running and benchmark it.

---

## Iteration 2

## STATE
Two data points: baseline at 2740.78 µs, one failed SDPA attempt at 5022 µs. The SDPA failure is well-understood: expanding K and V to `[128, 128, kv_len, 576]` allocates ~300 MB of new tensor per call, completely overwhelming memory bandwidth. The baseline remains the best result. The key insight from experiment #2 is that **the absorption trick must not be broken** — the whole point is that K and V are never expanded per-head; instead, `q_nope` is projected into the latent space, keeping K as `[bs, kv_len, dkv]` (single "head"). This is a multi-head-query, single-head-KV attention — a form of GQA/MQA where all 128 query heads share the same K/V.

## RATIONALE
The correct formulation is: all 128 query heads share the same K (`kv_nope`, shape `[bs, kv_len, dkv=512]`) and the same K_rope (`[bs, kv_len, 64]`). This is equivalent to MQA (1 KV group). PyTorch SDPA supports GQA natively when you pass K/V with `n_kv_heads=1` and Q with `n_heads=128` — you don't need to expand. The result is `[bs, nh, 1, dkv]` which is then projected per-head through `wV`. This single-KV-group layout means the attention is cheap on the K/V side (low bandwidth) and the SDPA kernel handles the `nh=128` Q heads against 1 KV head efficiently. Additionally, the score is `q_nope_latent · kv_nope + q_rope · k_rope` — but with only 1 KV group for both terms, SDPA can handle the composite key directly without any expansion.

## PROPOSAL

**Use SDPA correctly in MQA/GQA mode — 1 KV group, 128 Q heads, no expansion.**

1. **Keep K and V at 1 KV head**: shape `[bs, 1, kv_len, head_dim]` rather than `[bs, nh, kv_len, head_dim]`. PyTorch's SDPA with `enable_flash=True` supports GQA where `n_heads_q` is a multiple of `n_heads_kv`. Pass Q as `[bs, nh, 1, comp_dim]` and K/V as `[bs, 1, kv_len, comp_dim]`.

2. **Composite key dimension**: use `dkv + d_rope = 512 + 64 = 576` as the effective head dim for attention, with Q being `[q_nope_latent | q_rope]` of dimension 576 and K being `[kv_nope | k_rope]` of dimension 576. This keeps a single contiguous K without any per-head expansion.

3. **V stays as kv_nope latent** at `[bs, 1, kv_len, dkv=512]`, and after attention returns `[bs, nh, 1, dkv]`, apply the per-head `wV` projection.

4. **Ensure all inputs to SDPA are contiguous** — the `cat` outputs may or may not be contiguous; call `.contiguous()` before the SDPA call to avoid hidden copy overhead.

The expected outcome: SDPA in MQA mode should be dramatically cheaper than the baseline's 3-pass (scores_nope + scores_rope + softmax), because (a) flash attention avoids materializing the full score tensor and (b) 1 KV group means K/V memory traffic is ~128× smaller than in experiment #2.

---

## Iteration 3

## STATE
Three experiments: baseline at 2740 µs, SDPA-expand at 5022 µs (discarded), SDPA-GQA OOM crash. The PyTorch SDPA path is now well-understood to be broken for this problem: head_dim=576 is well outside the supported range for FlashAttention (which requires head_dim ≤ 256 and typically powers of 2). The GQA mode materializes `128 × 1 × kv_len × 576` intermediates at the math backend level — 144 GiB at kv_len=4096. This entire SDPA path is exhausted. The baseline remains best.

## RATIONALE
The root cause of all SDPA failures is head_dim=576, which exceeds FlashAttention's hardware limits. The correct fix is to **decompose the attention computation into two separate standard-sized attention calls**: one for the nope component (head_dim=512) and one for the rope component (head_dim=64), adding their score contributions before softmax. But neither 512 is a great flash-attention head_dim either. The more tractable path is to go back to basics and recognize what's actually expensive in the baseline: (1) the einsum `bhd,hdk->bhk` for `q_nope_latent` is a `[128,128,128]×[128,128,512]` batched matmul — expensive; (2) the two separate score matmuls + add; (3) the Triton softmax (3 passes over 8192 columns); (4) the V-aggregation matmul; (5) the final einsum for wV. A pure Triton fused kernel can handle all of steps 2–4 together (compute scores_nope + scores_rope in a single tile, do online softmax, accumulate V) while keeping the per-head K/V as shared across the batch. This avoids the head_dim constraint entirely.

## PROPOSAL

**Write a custom Triton fused attention kernel that handles the split-score architecture directly.**

The key insight: the attention scores are `scores = (q_nope_latent @ kv_nope^T + q_rope @ k_rope^T) * scale`. Both K components (`kv_nope` and `k_rope`) have 1 "group" shared by all 128 query heads. A Triton kernel can:

1. **Tile over the kv_len dimension** (the standard flash-attention tiling strategy — load a block of K/V from HBM once per tile)
2. **Compute both score contributions per tile**: `q_nope_latent_h · kv_nope_tile^T + q_rope_h · k_rope_tile^T` — this is two small dot products per element, not per-head expansion
3. **Maintain online softmax state** (running max and sum) across tiles — classic flash-attn numerics
4. **Accumulate the V output** (from `kv_nope_tile`) into a per-head running sum in SRAM

The grid should be `(bs × nh,)` with each program handling one (batch, head) pair. Each program streams through kv_len in tiles. The Q vectors for each head are small (512+64=576 floats) and fit entirely in registers/SRAM. The K/V tiles are shared across all heads within a batch (since there's only 1 KV group) — but since each Triton program handles one head independently, each just loads from the same kv_nope/k_rope base pointer. Memory coalescing is excellent: sequential access along the kv_len dimension.

This approach avoids materializing the full score tensor, avoids the head_dim limit of FlashAttention, and should achieve near-roofline performance on the attention phase. Keep the GEMM phases (wDQ, wDKV, wUQ, wO, q_nope→latent, M→output) as standard `F.linear` / `torch.matmul` calls for now — those are compute-bound and cuBLAS handles them well.

---

## Iteration 4

## STATE
Four experiments: baseline holds at 2740 µs. The custom Triton flash-attn kernel was catastrophically slow at 95,770 µs — the issue is clear from the code: each of the 128×128=16,384 programs independently loads the *entire* 512D K/V vectors in a serial loop, using element-wise multiply+sum that degenerates to scalar SIMT operations with massive register pressure. The `tl.sum(q[None,:] * kv_tile, axis=1)` pattern with DKV=512 means each thread holds 512 float32 registers — this is register-spill territory. The grid is also wrong for occupancy at this scale.

## RATIONALE
The Triton approach failed because of the wrong parallelism grain: one program per (batch, head) forces 16,384 serial loops over kv_len with 512-wide vector ops that don't map to GPU SIMT efficiently. The correct approach is to reshape the entire attention computation as standard **batched matrix multiplications** that cuBLAS can execute optimally, eliminating all custom kernels for the attention core. The key insight: `scores_nope = q_nope_latent @ kv_nope^T` is a batched GEMM `[bs, nh, dkv] × [bs, dkv, kv_len]` — but since all heads share the same KV, this is really `[bs×nh, dkv] × [bs, dkv, kv_len]` which can be computed as `[bs, nh, dkv] × [bs, dkv, kv_len]` via `torch.bmm` after reshaping. The Triton softmax in the baseline makes 3 passes over 8192 columns — replacing with `torch.softmax` in fp32 might actually be faster given cuDNN's fused kernel. The real win is eliminating the `einsum` calls (which emit suboptimal code) and replacing with explicit `torch.bmm` / `matmul` with correct shapes and contiguous layouts.

## PROPOSAL

**Optimize the baseline's PyTorch ops directly — no custom kernels. Focus on eliminating einsum and fixing tensor layouts for cuBLAS.**

The current baseline spends time on:
1. `torch.einsum('bhd,hdk->bhk', q_nope, wK)` — this is a batched matmul but einsum may choose a suboptimal path. Replace with explicit `torch.bmm` after reshaping: `q_nope.view(bs*nh, d_nope) @ wK.view(nh*d_nope, dkv)` — or better, `(q_nope.reshape(bs, nh, d_nope) @ wK.permute(0,2,1))` directly.
2. The split into `scores_nope` and `scores_rope` as two separate matmuls — these could potentially be merged but the real issue is the intermediate `[bs, nh, kv_len]` score tensors being allocated twice.
3. The Triton softmax being 3-pass — replace with `F.softmax(..., dim=-1)` which uses cuDNN's fused kernel.
4. `torch.matmul(attn, kv_nope_input)` — `attn` is `[bs, nh, kv_len]` and `kv_nope_input` is `[bs, kv_len, dkv]`. This is a standard batched GEMM but the broadcasting over nh on the KV side may be suboptimal. Reshape to `[bs*nh, kv_len] @ [bs, kv_len, dkv]` using `expand+reshape` only for the attention weights (small), keeping kv_nope as shared.
5. `einsum('bhd,hdk->bhk', M, wV_T)` — same issue as #1, replace with explicit batched matmul.

The goal: clean PyTorch-native code using `torch.bmm`/`matmul` with properly shaped contiguous tensors, and `F.softmax` instead of the custom Triton softmax. This removes overhead from einsum dispatch and Triton kernel launch latency, and lets cuBLAS choose optimal algorithms for each GEMM.

---

## Iteration 5

## STATE
Five experiments, baseline still best at 2740 µs. A clear pattern has emerged: **any attempt to expand K/V per-head OOMs at benchmark sequence lengths.** The 3 crashes/discards all stem from materializing `[bs, nh, kv_len, dim]` tensors. The Triton 1-program-per-head approach failed due to register pressure with DKV=512. The worker keeps gravitating toward bmm/expand approaches that OOM. We need to change direction entirely.

## RATIONALE
The fundamental constraint is: K/V must stay as `[bs, kv_len, dkv]` (no per-head expansion). The baseline already does this correctly with `torch.matmul(q_nope_latent, kv_nope_T)` — it broadcasts over the `nh` dimension naturally. The baseline's matmul `[bs, nh, dkv] @ [bs, dkv, kv_len]` is valid PyTorch broadcasting and doesn't expand memory. The baseline is actually structurally correct — the issue is execution efficiency, not structure. The real question is: **what is actually slow in the 2740 µs baseline?** The answer must come from understanding the op breakdown. Given the SOL of ~245 µs, the baseline is ~11× over. The dominant costs are likely: (1) the large GEMMs — `wDQ` is `[7168→1536]`, `wUQ` is `[1536→128×192]`, `wO` is `[128×128→7168]`; (2) the k_rope RoPE computation over 8192 positions; (3) the score matmuls. A completely different strategy: **skip the q_nope absorption trick entirely and instead compute attention directly in the original head space**, using `xformers` memory-efficient attention which supports arbitrary head dims without OOM.

## PROPOSAL

**Use `xformers.ops.memory_efficient_attention` with the split-score computed as a single additive bias.**

The xformers MEA kernel accepts an `attn_bias` argument that is added to the pre-softmax scores. This gives us the exact escape hatch needed for the split-score architecture:

1. Use the rope component as the primary attention with standard head_dim=64: Q_rope `[bs, 1, nh, d_rope]`, K_rope `[bs, kv_len, 1, d_rope]` — xformers MEA handles GQA-style broadcasting natively and supports head_dim=64 within FlashAttention's constraints.

2. Compute `scores_nope = q_nope_latent @ kv_nope^T` as a standard `[bs, nh, kv_len]` matmul (no expansion), then pass this `[bs, nh, 1, kv_len]` tensor as the `attn_bias` to xformers MEA. This adds the nope contribution to the rope scores before softmax — mathematically equivalent to the original split-score sum.

3. The V for xformers MEA is `kv_nope` at `[bs, kv_len, 1, dkv]` — single KV head. xformers will handle the GQA (1 KV group, 128 Q heads) correctly since dkv=512 fits within its supported head_dim range when V is separate from K.

4. **Fallback**: if xformers is unavailable or doesn't support these shapes, fall back to the baseline's matmul approach but with `F.softmax` (float32, dim=-1) replacing the Triton softmax, and ensure all score/attn tensors are float32 to avoid precision issues.

This approach never expands K/V per-head, uses hardware-optimized fused attention for the dominant cost, and correctly handles the split-score via the bias mechanism.

---

## Iteration 6

## STATE
Six experiments, baseline remains best at 2740 µs. The xformers path (exp 6, 4741 µs) confirms that any path involving mismatched Q/V head dims forces a non-FlashAttention fallback that's slower than baseline. All fused-attention approaches have failed due to either OOM (head_dim too large) or fallback to slow paths. The Triton per-head kernel was 35× slower due to register pressure. The problem is structurally resistant to standard flash-attention kernels: K/V head_dim=512 is too large, and the split-score architecture (nope + rope) doesn't map cleanly to any existing fused kernel.

## RATIONALE
It's time to revisit what's actually taking 2740 µs and find wins that are definitively achievable. The baseline structure is: `wDQ` GEMM → `wDKV` GEMM → `wUQ` GEMM → q_nope_latent einsum → 2 score matmuls → softmax → V-agg matmul → wV einsum → `wO` GEMM. The score matmuls are `[bs,nh,dkv]@[bs,dkv,kv_len]` and `[bs,nh,d_rope]@[bs,d_rope,kv_len]^T` — these use PyTorch's broadcasting matmul which should be fine. But the **einsum calls are suspicious**: `torch.einsum('bhd,hdk->bhk', q_nope, wK)` is `[128,128,128]×[128,128,512]` — this is a contracted product that einsum may route through a slow path. More importantly, the two score matmuls should be **the same operation** and can potentially be merged: combine `q_nope_latent` and `q_rope` along a new dimension to form a combined Q, and `kv_nope` and `k_rope` into a combined K, and do a **single** batched matmul. This halves kernel launches and improves arithmetic intensity. Also: `F.softmax` in fp32 is likely faster than the 3-pass Triton softmax for `n_cols=8192`.

## PROPOSAL

**Make the minimal targeted improvements to the baseline that avoid all OOM risks:**

1. **Replace the two separate score matmuls with a single fused matmul** by concatenating the queries and keys along the head_dim axis. After computing `q_nope_latent` (`[bs, nh, dkv]`) and `q_rope` (`[bs, nh, d_rope=64]`), stack them as `Q_combined = [q_nope_latent | q_rope]` of shape `[bs, nh, dkv+d_rope=576]`. Similarly stack `K_combined = [kv_nope | k_rope]` as `[bs, kv_len, 576]`. Then `scores = Q_combined @ K_combined^T` in a **single** batched matmul that uses PyTorch's broadcasting (no per-head expansion). This gives `[bs, nh, kv_len]` scores correctly via the broadcasting over the batch×head dims against the single-group KV.

2. **Replace the Triton 3-pass softmax with `F.softmax(..., dim=-1)`** computed in float32 on the `[bs, nh, kv_len]` score tensor directly (no reshaping needed, PyTorch handles the last-dim softmax efficiently).

3. **Replace the `einsum('bhd,hdk->bhk')` calls with `torch.matmul`** after ensuring shapes are properly arranged: for `q_nope_latent`, use `torch.matmul(q_nope, wK.transpose(-2,-1))` where `wK` is `[nh, d_nope, dkv]` — this is a direct `[bs, nh, d_nope] @ [nh, dkv, d_nope]^T` which broadcasts correctly. For the wV application, similarly use direct matmul.

4. **Ensure `kv_nope_input` and `k_rope` are taken as views of the already-contiguous `kv_lora` tensor without extra `.contiguous()` copies** — the `kv_lora` from the cache is already laid out as `[bs, kv_len, dkv+d_rope]`, so slicing along the last dim gives non-contiguous views that force implicit copies in matmul. Pre-computing a single concatenated `K_combined` copy of shape `[bs, kv_len, 576]` at the slice step is cheaper than having two separate copies later.

---

## Iteration 7

## STATE
Seven experiments, baseline still holds at 2740 µs. The full picture is now clear: exp 7 reveals that `torch.cat([kv_nope, k_rope])` over a `[128, kv_len, 576]` tensor costs ~10 ms alone — the KV cache data movement is the dominant bottleneck. Every approach that touches the KV data more than once is penalized. The baseline's two-separate-matmuls structure is actually close to optimal for the score computation because it reads `kv_nope` and `k_rope` exactly once each. The problem is not the number of matmul kernel launches — it's raw memory bandwidth.

## RATIONALE
The 2740 µs baseline is dominated by reading the KV cache: `kv_nope [128, kv_len, 512]` + `k_rope [128, kv_len, 64]` = `128 × 6144 × 576 × 2 bytes ≈ 904 MB` per pass, and it's read twice (once for scores, once for V-agg). At H200's ~3.35 TB/s bandwidth, that's ~270 µs minimum just for KV reads — which is actually close to the SOL. The k_rope RoPE computation over all kv_len positions is unnecessary: `k_rope` does not need to be fully materialized since we only ever use it to compute `scores_rope = q_rope @ k_rope^T` — we can fuse the RoPE into the score computation, avoiding a full `[128, kv_len, 64]` materialization. Similarly, `scores_rope` and `scores_nope` are both intermediates feeding into the same softmax. The real question is: **can we avoid the `.contiguous()` copies** that the baseline makes, and **can we avoid computing k_rope as a separate tensor** by fusing RoPE into the score kernel?

## PROPOSAL

**Focus on eliminating unnecessary memory allocations and copies within the baseline structure — no new GEMM restructuring.**

The specific wins to target, in order of estimated impact:

1. **Eliminate the `kv_nope_input.contiguous()` copy**: `kv_lora[..., :dkv]` is a non-contiguous slice (stride on last dim = dkv+d_rope). The baseline calls `.contiguous()` which copies `128 × kv_len × 512 × 2 bytes ≈ ~800 MB`. Instead, the score matmul `q_nope_latent @ kv_nope^T` and the V-aggregation `attn @ kv_nope` can work with the non-contiguous slice directly — PyTorch matmul handles non-contiguous inputs by internally doing the right thing, or the worker can try `torch.as_strided` to avoid the copy.

2. **Eliminate the k_rope materialization**: instead of `k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k` as a full `[bs, kv_len, 64]` tensor, directly compute `scores_rope` via a small custom Triton kernel that fuses RoPE application + dot product in a single pass. Grid `(bs,)`, each program handles all kv_len positions for one batch item, computing the `[nh, kv_len]` rope scores without ever writing the full rotated `k_rope` to HBM.

3. **Replace the 3-pass Triton softmax with `F.softmax`** on the `[bs, nh, kv_len]` scores tensor (no reshape needed). The baseline reshapes to `[bs*nh, kv_len]` for the Triton kernel — `F.softmax` handles 3D tensors natively on the last dim.

4. **Ensure `kv_lora` slices avoid copies**: if the KV cache stores data as `[bs, kv_len, dkv+d_rope]`, then reading `kv_lora[..., :dkv]` and `kv_lora[..., dkv:]` as non-contiguous views is fine for matmul (PyTorch will handle it) but saves two large copy operations.

The worker should implement these as incremental changes to the baseline, verifying each step is memory-safe (no expand/reshape that forces materialization).

---

## Iteration 8

## STATE
Eight experiments, baseline still holds at 2740 µs. Experiment 8 (3809 µs) confirms an important finding: non-contiguous KV slices for matmul are *slower* than explicit `.contiguous()` copies — cuBLAS requires contiguous layout and PyTorch's implicit copy-on-demand is slower than an upfront copy. The F.softmax fp32 path also appears to be slower than the baseline's bfloat16 Triton softmax. We have now exhausted: flash-attention variants (OOM/wrong head dim), Triton per-head kernel (register pressure), bmm expansion (OOM), xformers (fallback path), combined-K matmul (extra copy cost), and targeted copy elimination (backfires). Every approach has been slower.

## RATIONALE
The pattern across 7 failed experiments is clear: this problem is extremely sensitive to memory allocation, and the baseline is already doing the right structural things — it's just not fast enough. The only remaining lever is the **GEMM phase**. Looking at the compute budget:
- `wDQ`: `[128, 7168] → [128, 1, 1536]` = 128×7168×1536 FLOPs ≈ 1.4 GFLOP
- `wUQ`: `[128, 1536] → [128, 128×192]` = 128×1536×24576 FLOPs ≈ 4.8 GFLOP  
- `wO`: `[128, 128×128] → [128, 1, 7168]` = 128×16384×7168 FLOPs ≈ 15 GFLOP

At H200's ~989 TFLOPS BF16, these GEMMs take ~20 µs combined. The attention phase (score matmuls + V-agg) over kv_len=4096-6144 accounts for most of the remaining 2740 µs. The key insight we haven't tried: **precomputing `wK` absorbed into `wUQ`** to eliminate the `q_nope→q_nope_latent` einsum, and **fusing `wV` with `wO`** to reduce the output projection overhead. More critically: the `q_nope_latent` einsum `[bs,nh,d_nope] @ [nh,d_nope,dkv]` = `[128,128,128] @ [128,128,512]` is computed per-call but `wK` is a fixed weight — this should be pre-fused into `wUQ` offline so it's never computed at runtime.

## PROPOSAL

**Pre-absorb the weight matrices offline to eliminate runtime projections.**

The key algebraic identity: `q_nope_latent = q_nope @ wK = (q_lora @ wUQ_nope) @ wK = q_lora @ (wUQ_nope @ wK)`. Since `wUQ_nope` (`[dq, nh*d_nope]`) and `wK` (`[nh, d_nope, dkv]`) are both fixed weights, their product `wUQ_absorbed = wUQ_nope @ wK` (shape `[dq, nh*dkv]`) can be precomputed once and cached. Then:
- Instead of two steps (wUQ→q_up→q_nope→q_nope_latent), do one GEMM: `q_nope_latent = q_lora @ wUQ_absorbed` directly.
- Similarly, `wV` and `wO` can be fused: `y = M @ wV @ wO` = `M @ (wV @ wO)` where the product is precomputed.

This eliminates the `einsum('bhd,hdk->bhk')` operations entirely (replaced by a single larger GEMM), reduces kernel launches, and improves arithmetic intensity. The absorbed weight `[dq=1536, nh*dkv=128*512=65536]` is large but computed only once at startup. This is the standard DeepSeek MLA optimization used in production.

The worker should:
1. Cache `wUQ_absorbed = (wUQ[:, :nh*d_nope].reshape(dq, nh, d_nope) @ wK).reshape(dq, nh*dkv)` on first call
2. Compute `q_nope_latent = F.linear(q_lora.squeeze(1), wUQ_absorbed).view(bs, nh, dkv)` directly  
3. Keep the rope part of wUQ separate: `q_rope = F.linear(q_lora.squeeze(1), wUQ_rope).view(bs, nh, d_rope)`
4. Similarly cache `wVO_absorbed = (wV.reshape(nh*dv, dkv) ... @ wO)` for the output path

---

## Iteration 9

## STATE
Nine experiments, baseline still best at 2740 µs. Experiment 9 (4081 µs) definitively shows that weight absorption makes things *slower* — the absorbed weights `[1536, 65536]` and `[65536, 7168]` create GEMMs that are too large and poorly shaped for cuBLAS. We've now exhausted all major algorithmic directions: flash attention (OOM/head_dim), Triton fused attention (register pressure), per-head expansion (OOM), copy elimination (slower), K concatenation (slower), weight absorption (slower). The baseline is proving remarkably robust.

## RATIONALE
It's time to stop trying to restructure the computation and instead focus on what the baseline is actually bottlenecked on. Looking at experiment 8 (3809 µs) vs the baseline (2740 µs): the only differences were non-contiguous KV slices and F.softmax. The non-contiguous slices hurt. But here's what we haven't tried: **keeping the baseline structure exactly** but just removing the one component that might be most wasteful — the `k_rope` materialization. The baseline computes `k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k` which allocates a full `[128, kv_len, 64]` BF16 tensor. This is only ~100 MB but requires 4 element-wise ops over `kv_len` positions. The `_rotate_half` itself allocates a temporary via `torch.cat`. However the real opportunity is the **scores_rope computation**: `q_rope @ k_rope.T` where `k_rope` is `[bs, kv_len, 64]`. Since we *know* `k_rope[b, t, :] = rot(k_rope_input[b, t, :], cos_table[t], sin_table[t])` and `q_rope[b, h, :]` is a single vector, we can compute `scores_rope[b, h, t] = q_rope[b,h,:] · rot(k_rope_raw[b,t,:], t)` without ever materializing rotated `k_rope`. This saves one full `[128, kv_len, 64]` allocation and the associated bandwidth, and can be done via a simple Triton kernel that computes rope-rotated dot products directly. This is achievable, memory-safe, and targets a real bottleneck.

## PROPOSAL

**Fuse the RoPE rotation with the rope score computation in a single Triton kernel — eliminating the `k_rope` materialization.**

Currently the baseline does:
1. `k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k` → allocates `[128, kv_len, 64]`  
2. `scores_rope = q_rope @ k_rope.T` → reads that tensor back

Propose replacing both with a single kernel that computes `scores_rope[b, h, t]` directly from `k_rope_input[b, t, :]` (the un-rotated key in the KV cache) and `cos_table[t]`, `sin_table[t]`. 

The kernel design: grid over `(bs, nh)`, each program computes the full `kv_len` score vector for one (batch, head) pair. For each position `t`, load `k_rope_raw[b, t, :]` (64 dims), load `cos_table[t]` and `sin_table[t]` (64 dims each), apply rotation inline, compute dot with `q_rope[b, h, :]`, and write one scalar to `scores_rope[b, h, t]`. 

This kernel:
- Eliminates the `[128, kv_len, 64]` k_rope tensor (saves ~100 MB allocation + write bandwidth)
- Reduces HBM reads: `k_rope_raw` is already being read for the nope scores via `kv_lora[..., dkv:]`, so accessing it here amortizes the cache-line fetch
- The 64-dim rope dot product is tiny and fits entirely in registers; the loop over kv_len is the outer dimension with good memory access patterns

Keep everything else identical to the baseline (contiguous kv_nope, Triton softmax, einsum projections).

---

## Iteration 10

## STATE
Ten experiments, zero improvements. The baseline holds at 2740 µs. The complete picture: every custom Triton kernel is dramatically slower (35× to 2.5×) due to serial loops over kv_len with bs×nh=16,384 programs; every per-head K/V expansion OOMs; every alternative to the baseline's matmul structure is slower. The baseline's PyTorch matmul path with cuBLAS broadcasting appears to be near-optimal for what the GPU can do given this problem structure.

## RATIONALE
We need to step back and consider what the scoring baseline is *actually doing* at the hardware level. The baseline's key operations are all memory-bandwidth-bound reads of the KV cache. At ~2740 µs on H200, we're reading `~1.5 GB` of KV data (twice — once for scores, once for V-agg) at ~3.35 TB/s theoretical bandwidth — giving ~900 µs just for the KV reads, suggesting significant overhead elsewhere. The `_rotate_half` function uses `torch.cat` which is particularly expensive — it allocates a new tensor and is not in-place. The `k_rope` computation `k_rope = k_rope_input * cos_k + _rotate_half(k_rope_input) * sin_k` calls `torch.cat` inside `_rotate_half`, creating an extra allocation. Replacing `_rotate_half` with an in-place-friendly implementation using `torch.empty` + `torch.split` would avoid the extra allocation overhead. More critically, the entire k_rope computation at `[128, kv_len, 64]` could use `torch.ops.aten` to avoid the cat overhead. However, at this point the most productive path may be to simply **try the pure baseline code but profile where the time actually goes** — specifically checking if the kv_nope `.contiguous()` copy is the single largest bottleneck, and whether the `scores_nope` matmul's broadcasting is triggering a slow cuBLAS path.

## PROPOSAL

**Try bypassing the `.contiguous()` copy for `kv_nope_input` by making the kv_lora slice directly usable in matmul — but using `torch.as_strided` or storing a transposed copy instead.**

The key insight we haven't exploited: `kv_nope_input.transpose(1,2)` currently produces `[bs, dkv, kv_len]` from a non-contiguous `[bs, kv_len, dkv]`. The `.contiguous()` before creates a `[bs, kv_len, dkv]` copy with row-major layout, then the `.transpose(1,2)` makes it non-contiguous again. Instead, **store a pre-transposed copy** `kv_nope_T` directly from the KV cache slice, avoiding the intermediate copy. Additionally:

1. Replace the 3-pass Triton softmax with PyTorch's `torch.softmax` in **bfloat16** (not float32 — experiment 8 used float32 cast which adds conversion overhead). Test if bfloat16 softmax is faster.

2. Replace `_rotate_half` with a non-allocating version: `torch.roll(x, x.shape[-1]//2, dims=-1)` followed by sign flip on the first half — this avoids the `torch.cat` allocation.

3. For the `k_rope` computation, apply the rotation *in-place* on a freshly-allocated tensor rather than creating a new one: allocate `k_rope = torch.empty_like(k_rope_input)` then fill it with the rotated values in two operations.

Keep all score matmuls and einsum projections identical to the baseline. The goal is to reduce the number of temporary allocations (especially from `torch.cat` in `_rotate_half`) and test if bfloat16 softmax is viable.

---

## Iteration 11

## STATE
Eleven experiments, zero improvements. The baseline at 2740 µs is completely impenetrable with every optimization direction we've tried. Experiment 11 confirms that `torch.roll` + sign-flip is *slower* than `torch.cat` for rotate_half, and BF16 softmax is slower than the Triton 3-pass kernel. We have now confirmed that literally every individual component of the baseline is already near-optimal: contiguous copies, einsum projections, Triton softmax, k_rope materialization, and separate score matmuls. Every structural change makes things worse.

## RATIONALE
At this stage, we need to accept that Python-level optimization of this pipeline has hit a wall. The baseline is 2740 µs vs a ~245 µs SOL — an 11× gap — but **we can't close it with PyTorch-level changes**. The fundamental bottleneck is that reading and processing the KV cache data (`[128, kv_len, 576]` BF16) through multiple serial kernel passes accumulates memory bandwidth costs that no Python-level restructuring can eliminate without a fundamentally different execution model. The one direction we haven't tried seriously is **CUDA graph capture** — wrapping the entire forward pass in a CUDA graph eliminates Python overhead and kernel launch latency, which for a pipeline of ~10+ kernel calls can be substantial (each CUDA kernel launch has ~5-10 µs overhead on the CPU side). With 10+ operations, this could account for 50-100 µs. It won't close the full gap, but it's one of the few remaining levers that doesn't change the mathematical structure.

The second untried direction: **`torch.compile`** — PyTorch's JIT compiler can fuse elementwise operations (like the RoPE computation) into a single kernel, eliminating the intermediate `k_rope` materialization that our Triton attempt failed to do because of the serial-loop problem. `torch.compile` with `mode='reduce-overhead'` applies CUDA graph capture automatically, and `mode='max-autotune'` can discover optimal cuBLAS algorithms. This is a high-confidence direction because it requires minimal code change and directly targets the multi-kernel overhead.

## PROPOSAL

**Apply `torch.compile` to the custom kernel function (or key sub-functions) with `mode='reduce-overhead'` or `'max-autotune'`.**

The worker should:

1. **Wrap `custom_kernel` with `torch.compile`** using `mode='reduce-overhead'` — this enables CUDA graph capture for the entire forward pass, eliminating Python interpreter overhead and GPU kernel launch latency across all sequential operations. The baseline has approximately 15+ sequential GPU operations (GEMMs, elementwise RoPE, matmuls, softmax, etc.), each with launch overhead.

2. If full function compilation hits issues (e.g., due to the `kv_cache(kv_lora_input)` call modifying in-place state), then **compile only the compute-intensive sub-portion**: the part after the KV cache update (i.e., from the query projections through to the output), leaving the KV cache call outside the compiled region.

3. Use the **baseline code exactly** as the body — no structural changes. `torch.compile` should automatically fuse the elementwise RoPE operations, potentially eliminate the `k_rope` intermediate, and select optimal cuBLAS algorithms.

4. Add `torch.compiler.disable` or dynamic shape handling if needed to avoid recompilation across different `kv_len` values at correctness-test vs benchmark time.

The expected gain is 100-400 µs from eliminating launch overhead and fusing elementwise operations, bringing us from 2740 µs toward ~2300-2600 µs as a first step.

