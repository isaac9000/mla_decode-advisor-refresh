# Advisor Proposals

---

## Iteration 1

## STATE
Only one data point exists: the baseline at 2739.78 µs, which is ~13× slower than the roofline SOL (~210–281 µs). The baseline uses PyTorch F.linear for all GEMMs plus naive Triton softmax and RoPE kernels. There is no fusion, no Flash Attention, and the attention phase materializes full [bs=128, nh=128, kv_len] score tensors in fp32/bf16 before softmax — this is the dominant bottleneck at large sequence lengths.

## RATIONALE
The attention phase — `Q @ K^T` → softmax → `@ V` — operates on a [128, 128, 4096/6144] score matrix. At seq_len=4096, that's 128×128×4096 = 67M elements just for the score tensor, which must be written and re-read twice (score → softmax → attn×V). Flash Attention's tiled approach eliminates this intermediate materialization entirely, reducing HBM traffic by >10×. The LoRA projection GEMMs and output projection are also entirely unfused. The biggest single leverage point is replacing the naive attention with a fused Flash-Attention-style kernel via `torch.nn.functional.scaled_dot_product_attention` (xformers/cuDNN path) or a Triton flash-attention kernel. This alone should drop latency by 5–10× on the attention phase.

The absorption trick is already applied (q_nope_latent = q_nope @ wK, then scores_nope = q_nope_latent @ kv_T) — this is good. The remaining wins are: (1) fused attention, (2) better GEMM dispatch.

## PROPOSAL

Replace the three-step attention (score materialization → Triton softmax → value aggregation) with `torch.nn.functional.scaled_dot_product_attention` (SDPA). PyTorch's SDPA dispatches to cuDNN Flash Attention or the efficient-attention backend automatically, fusing the QK^T, softmax, and AV operations into a single tiled kernel with O(seq_len) HBM traffic instead of O(seq_len²). 

Concretely: reshape the "nope-absorbed" query and the RoPE query into head-batched tensors of shape [bs, nh, 1, head_dim], reshape the key and value tensors similarly into [bs, nh, kv_len, head_dim], and pass them to `F.scaled_dot_product_attention`. This requires materializing the per-head K and V only once (they are derived from kv_nope via wK/wV projections), but the attention computation itself will be memory-efficient. The nope and rope components need to be concatenated along the head_dim axis to form a single Q/K pair before passing to SDPA, or alternatively handled as two separate SDPA calls whose logits are summed — but the cleanest approach is to concatenate [q_nope_latent, q_rope] and [k_nope_latent, k_rope] along the head_dim axis so a single SDPA call handles the full score. This eliminates both the Triton softmax and the explicit score tensor.

---

## Iteration 2

## STATE
Two experiments so far: baseline at 2739.78 µs, SDPA attempt at 4736 µs (discarded). The SDPA attempt failed because the MLA absorbed formulation keeps K as the shared latent `kv_nope` tensor (shape [bs, kv_len, 512]) — not a per-head contiguous tensor — so SDPA falls back to a slow memory-inefficient path. The roofline SOL is ~211–281 µs, leaving ~10× headroom. The primary bottleneck is clear from the math: at kv_len=4096, the score tensor `[128, 128, 4096]` = 67M elements written twice through HBM.

## RATIONALE
The worker's experiment report gives critical diagnostic information: "contiguous K=[bs,nh,kv_len,576] would be ~137GB" — that's the per-head K tensor at full dim (576). But the actual absorbed key is only 512-dim (kv_lora_rank), giving a contiguous K at [128, 128, 4096, 512] ≈ 68GB — still far too large. The key insight is that in the absorbed MLA formulation, **K and V are shared across all 128 heads** (they're just `kv_nope`, shape [bs, kv_len, 512]). This is the structure Flash Attention doesn't natively exploit — but a custom Triton MLA-decode kernel *can* exploit it, keeping the shared KV in SRAM and processing multiple heads simultaneously.

The correct path forward is a **custom Triton flash-attention kernel for MLA decode** that takes advantage of the shared KV structure:
- Q per head: [bs, nh, 1, dkv] (nope absorbed) + [bs, nh, 1, d_rope]  
- K shared: [bs, kv_len, dkv] (not per-head)
- V shared: derived from the same latent via wV absorption

This is exactly the "MLA absorption" kernel structure that allows tiling over kv_len without ever materializing per-head K/V, keeping only one tile of the shared kv_nope in SRAM at a time and computing scores for all heads simultaneously.

## PROPOSAL

Implement a **custom Triton flash-decoding kernel** that exploits the MLA shared-KV structure. The kernel should tile over `kv_len` in the outer loop, and for each tile:
1. Load a tile of `kv_nope` (shared across heads) from HBM into SRAM — one load, used by all heads
2. Compute the nope score contribution for all heads in the tile: `q_nope_latent[h] @ kv_tile^T` — pure SRAM matmul
3. Load the RoPE key tile `k_rope` and compute the rope score contribution similarly
4. Accumulate running softmax statistics (online softmax, Flash-Attention style) in registers
5. Accumulate the value contribution `attn_weights @ kv_tile` in registers

The output accumulation in the value step also goes through `kv_nope` (same tensor, same tile), with the `wV` projection applied *after* accumulation (or fused into the tile). This keeps total HBM traffic at O(kv_len × dkv) for the shared KV read, eliminating the O(bs × nh × kv_len) score tensor write.

The key kernel design decision for the worker: process multiple heads per CTA by keeping the shared `kv_tile` in shared memory and iterating heads in registers, trading register pressure for reduced HBM traffic on the K read.

---

## Iteration 3

## STATE
Three experiments: baseline 2739.78 µs is the best. SDPA fell back to slow paths (4736 µs); custom Triton flash kernel was catastrophically slow (138K µs) due to register spilling from 512-dim accumulators and poor SM utilization (only 16K CTAs). Both custom kernel approaches failed at the first attempt. The baseline is still unbeaten, sitting ~13× from roofline. The key lesson from experiment #3: accumulating 512-dim vectors in registers per CTA is untenable — the standard flash-attention approach doesn't map cleanly to MLA's latent KV structure at these dimensions.

## RATIONALE
The worker's experiment reports contain critical insight: the baseline's *batched GEMM approach is actually reasonable* — the problem isn't the algorithmic structure, it's intermediate tensor allocation and kernel launch overhead. Looking at the baseline pipeline: it computes `scores_nope=[128,128,kv_len]` and `scores_rope=[128,128,kv_len]` separately, adds them, writes to HBM, then reads back for softmax. The score tensor at kv_len=4096 is 128×128×4096×2 bytes ≈ 134MB written + read = ~268MB total for just the score materialization, plus another ~134MB for softmax output. That's ~400MB of HBM traffic just in the attention score phase. The `torch.einsum` calls also launch unoptimized paths.

The most tractable improvement without a from-scratch custom kernel: **fuse the two score computations and softmax into a single pass using `torch.compile` or by restructuring the computation to avoid intermediate materializations**. Specifically, `scores_nope + scores_rope` writes a full [128,128,kv_len] tensor before softmax reads it — these can be fused. Also, `torch.einsum('bhd,hdk->bhk', q_nope, wK)` dispatches through a slow einsum path; replacing it with explicit `bmm` after transposition is measurably faster.

Critically, the value accumulation `M = attn @ kv_nope` produces [bs,nh,512] and then `einsum('bhd,hdk->bhk', M, wV_T)` projects to [bs,nh,128] — this second einsum can be fused: the projection `M @ wV_T` where wV_T is [nh,512,128] is a batched GEMM. Using `torch.bmm` explicitly with proper reshaping gives cuBLAS a better chance to pick an efficient kernel.

## PROPOSAL

Focus on **eliminating unnecessary intermediate tensor materializations and replacing einsum with explicit batched GEMM (bmm) calls** in the existing PyTorch pipeline — no custom Triton kernels yet.

Key transformations to implement:

1. **Replace `torch.einsum('bhd,hdk->bhk', q_nope, wK)` with `torch.bmm`**: reshape q_nope to [bs×nh, 1, d_nope] and wK to [bs×nh, d_nope, dkv] (broadcast), then bmm. Or better: treat this as a regular matmul with explicit broadcasting — `(q_nope @ wK)` where wK is reshaped to enable batched matmul with heads as the batch dimension.

2. **Replace the two separate matmuls + add + softmax chain** with a single fused operation: compute `scores = scores_nope + scores_rope` in-place without creating a third tensor, or compute them into a pre-allocated buffer.

3. **Replace the value einsum with bmm**: after computing `M = attn @ kv_nope` (shape [bs,nh,dkv]), apply `wV_T` via `torch.bmm(M.view(bs*nh, 1, dkv), wV_T.view(nh, dkv, dv).expand(bs*nh, dkv, dv))` — but use `torch.matmul` with proper broadcasting to avoid the expand copy.

4. **Use `torch.compile`** on the entire `custom_kernel` function with `mode="reduce-overhead"` to fuse elementwise ops (the score addition, scaling, RoPE multiply-add) and eliminate kernel launch overhead across the ~10 separate kernels in the baseline.

The goal is to beat 2739 µs through PyTorch-level optimization before attempting any new custom kernels.

---

## Iteration 4

## STATE
Four experiments with the baseline at 2739.78 µs still unbeaten. The last attempt (2842 µs, slightly worse) confirmed that PyTorch-level GEMM reshaping doesn't help and `torch.compile` crashes due to KV cache side effects. The score tensor materialization is still the clear bottleneck — ~400MB of HBM traffic just for the attention scores — but the approaches tried so far either crash, regress, or are catastrophically slow. The experiment record now shows: (1) SDPA doesn't work due to non-contiguous shared KV, (2) monolithic Triton flash kernels spill registers, (3) PyTorch-level ops are already near-optimal for their individual steps. The roofline gap remains ~13×.

## RATIONALE
The key architectural insight that's been missed: this is a **decode step with bs=128, sq=1**. The score computation produces [128, 128, 4096] tensors — that's 128 batch × 128 heads × 4096 seq tokens. But wait: `scores_nope = q_nope_latent @ kv_nope_T` has shapes [128, 128, 512] @ [128, 512, 4096] = [128, 128, 4096]. This is a batched GEMM where the batch=bs=128 and the "head" and "latent" dimensions are handled together. The actual dominant cost is this `[128, 128, 512] @ [128, 512, 4096]` matmul plus `[128, 128, 64] @ [128, 4096, 64]^T` for rope, and then `[128, 128, 4096] @ [128, 4096, 512]` for the value aggregation.

The most impactful single change not yet tried: **bypass the `kv_lora` dimension structure entirely by precomputing `wK` and `wV` absorbed into the query before the attention pass**, and then use `F.scaled_dot_product_attention` with *per-batch* (not per-head) grouping. Specifically: since kv_nope is [bs, kv_len, 512] and is shared across all 128 heads, we can treat it as a multi-query/grouped-query attention where K and V have 1 "head" and Q has 128 heads. SDPA with GQA (grouped query attention) in PyTorch ≥2.2 supports this via `enable_math=False` and the `query.expand` approach — but critically, if we use `torch.nn.functional.scaled_dot_product_attention` with Q=[bs,128,1,512] and K=[bs,1,1,512] this doesn't match. The right framing: treat the attention as bs=128 independent problems of shape [128 heads, 1, kv_len] score matrix, where K is the latent kv_nope shared across heads.

**The correct approach**: use `xformers.ops.memory_efficient_attention` or PyTorch's SDPA with the **GQA (grouped-query attention)** layout. Set up Q as [bs, nh, 1, dkv] (post-absorption) and K/V as [bs, 1, kv_len, dkv] — SDPA with `enable_math=False` should recognize that K/V has 1 head while Q has 128 heads and use the memory-efficient GQA path, reading KV only once from HBM and computing all 128 head scores per KV tile. This avoids both the HBM score tensor write and the 128× KV replication.

## PROPOSAL

Implement **GQA-style SDPA** by treating the MLA absorbed attention as grouped-query attention with 1 KV head and 128 Q heads. After computing `q_nope_latent` [bs, nh, dkv] and `q_rope` [bs, nh, d_rope], concatenate them along the head_dim axis to form Q=[bs, nh, 1, dkv+d_rope]. Construct K=[bs, 1, kv_len, dkv+d_rope] by concatenating kv_nope and k_rope (k_rope shape is [bs, kv_len, d_rope], but for GQA the K is per-token not per-head). Pass these to `F.scaled_dot_product_attention` — PyTorch ≥2.2 supports the case where `num_heads_q > num_heads_kv` (GQA) and will dispatch to an efficient kernel that broadcasts K/V across Q heads internally without materializing a per-head copy.

For the value side, V=[bs, 1, kv_len, dkv] (just kv_nope again), and the wV projection is applied after SDPA returns the [bs, nh, 1, dkv] context vector.

The key is that `F.scaled_dot_product_attention` with K/V having shape [bs, 1, kv_len, head_dim] and Q having shape [bs, nh, 1, head_dim] should trigger the GQA fast path (H100/H200 cuDNN Flash Attention v2 supports GQA natively), reading kv_nope only once from HBM for all 128 query heads. If the rope component creates a dimension mismatch, handle it by either (a) pre-adding the rope scores as a bias/mask to the SDPA attention bias parameter, or (b) computing the rope scores separately as a [bs, nh, 1, kv_len] bias tensor and passing it as `attn_mask` to SDPA.

---

## Iteration 5

## STATE
Five experiments, baseline at 2739.78 µs still best. The pattern is now clear: any approach that tries to fuse attention into a single kernel with the full dkv=512 V accumulator either OOMs (SDPA GQA materializes 137GB internally) or register-spills (Triton flash kernel). The baseline's score tensor [128,128,4096] at 134MB is the actual HBM bottleneck, but no standard path eliminates it without hitting the dkv=512 wall. The key constraint is crystallized: **the V-side output per head is 512-dimensional (dkv), not 128 (dv)** — wV projection only happens *after* the kv_nope weighted sum, so any flash-style accumulation must carry 512 floats per head, which is 8× larger than standard head_dim=64 flash attention.

## RATIONALE
The fundamental insight from experiment #5: SDPA with V=[bs,kv_len,512] OOMs because the output pre-projection is [bs,nh,kv_len,512]. The way out is to **apply the wV projection inside the attention loop** so we only ever accumulate 128-dimensional (dv) outputs, not 512. This is precisely what makes the custom kernel viable: if we can fuse `attn_weights @ kv_nope @ wV^T` into a single pass where the accumulator is [dv=128] per head (not [dkv=512]), register pressure drops 4× and the approach becomes feasible.

The concrete algebra: `output_h = sum_t(softmax_t * kv_nope_t) @ wV_h^T`. This can be rewritten as `sum_t(softmax_t * (kv_nope_t @ wV_h^T))` where `kv_nope_t @ wV_h^T` produces a scalar... wait, no — `kv_nope_t` is [dkv=512] and `wV_h` is [dv=128, dkv=512], so `kv_nope_t @ wV_h^T` = [dv=128]. So the accumulator becomes **[dv=128] per head**, not [dkv=512]. This is 4× smaller, bringing us from 2KB to 512B per thread — very manageable.

The Triton kernel from experiment #3 made the mistake of accumulating in dkv=512 space. The correct approach: precompute `v_t = kv_nope_t @ wV^T` as [bs, kv_len, nh, dv] (or equivalently use `wV` [nh, dkv, dv] to project each kv token), then accumulate [dv=128] per head with online softmax. However, precomputing [bs, kv_len, nh, dv] = [128, 4096, 128, 128] = 8B elements = 16GB — still too large. 

The correct fusion: in a tiled kernel, for each batch×head CTA, load a tile of kv_nope [tile_sz, 512], apply `wV_h` projection inline to get [tile_sz, dv=128], accumulate the weighted sum into [dv=128] registers. This keeps peak intermediate memory at just the tile, and the accumulator stays at 128 floats.

## PROPOSAL

Implement a **tiled Triton kernel that accumulates in dv=128 space** (not dkv=512), with grid = [bs, nh, num_kv_splits] for high SM utilization. For each CTA (one batch item, one head, one KV split):

1. Load `q_nope_latent_h` [dkv=512] and `q_rope_h` [d_rope=64] for this batch+head into shared memory
2. Load `wV_h` [dv=128, dkv=512] for this head into shared memory (this is 128KB — fits in SMEM)
3. Tile over the KV split: for each tile of `kv_nope` [tile_sz, 512] and `k_rope` [tile_sz, 64]:
   - Compute scores: `q_nope_latent @ kv_tile^T` + `q_rope @ k_rope_tile^T` → [tile_sz] scores
   - Project values: `kv_tile @ wV_h^T` → [tile_sz, dv=128] 
   - Accumulate online softmax with [dv=128] running sum (not [dkv=512])
4. Output [dv=128] context vector per (batch, head)

Optionally split over KV dimension and use a separate reduction pass (flash-decoding style) if the KV split parallelism is needed for SM utilization. The key correctness fix vs experiment #3 is the projection order: accumulate in dv=128 not dkv=512.

Worker should be careful about: (a) wV_h loading — it's 128×512×2 = 131KB per head, needs to fit in SMEM; (b) the tile size should be small enough that kv_tile stays in SMEM; (c) the grid should give enough CTAs (bs×nh×splits = 128×128×num_splits ≥ number of SMs on H200).

---

## Iteration 6

## STATE
Six experiments, baseline at 2739.78 µs still best. The critical constraint is now fully established: dkv=512 is too large for any register-based Triton accumulation (2KB/thread → severe spill), and SDPA-based approaches OOM at benchmark sizes because they materialize [bs,nh,kv_len,dkv] tensors. The baseline's matmul-then-project path is actually reasonable, but the dominant cost is the [128,128,4096] score tensor materialization and the intermediate [128,128,512] `M` tensor. Custom Triton flash kernels have all failed due to the high-dimensional latent space.

## RATIONALE
The history shows that the dkv=512 latent dimension blocks all flash-style fusions. But there's a completely different structural optimization that hasn't been tried: **pre-projecting the kv_nope into per-head K and V representations outside the attention loop** so that standard SDPA with proper head_dim=128 (not 512) can be used. Specifically: precompute `K_heads = kv_nope @ wK^T` → [bs, kv_len, nh, d_nope=128], and `V_heads = kv_nope @ wV^T` → [bs, kv_len, nh, dv=128]. These are standard head_dim=128 tensors that SDPA handles natively and efficiently. The cost of pre-projection: [128, 4096, 512] @ [512, 128×128=16384] for K = ~5B FLOPs, but the resulting K/V tensors are [128, 4096, 128, 128] = 8GB each — too large to materialize.

Wait — that's still too large. Let me rethink. The per-head K and V are [bs, kv_len, d_nope] and [bs, kv_len, dv] **per head**. With nh=128 heads, materializing all heads simultaneously requires [128, 128, 4096, 128] = 8GB. Still OOM.

The correct approach is **per-batch, sequential over heads** but that defeats parallelism. 

**The real opportunity**: the wUQ and wUKV projections are currently applied as two separate steps. The `q_nope` absorption into `q_nope_latent` via `wK` costs [bs, nh, d_nope] @ [nh, d_nope, dkv] = [128, 128, 128] @ [128, 128, 512] ≈ 16M ops. But this could be **fused with wUQ**: instead of computing `q_up = q_lora @ wUQ` (a [bs, dq=1536] @ [1536, nh*(d_nope+d_rope)] GEMM), we could compute `q_nope_latent` directly as `q_lora @ (wUQ_nope @ wK)` where the product `wUQ_nope @ wK` is a [dq=1536, nh*dkv] matrix precomputed once. This fuses two GEMMs into one, reducing HBM traffic for the query projection significantly.

## PROPOSAL

**Fuse the query projection chain into a single GEMM** by precomputing the absorbed weight matrix at initialization time (outside the hot path). Currently the code does: `q_lora → wUQ → q_up → q_nope → q_nope_latent (via wK)`. These are two sequential GEMMs: `[bs, 1536] @ [1536, nh*(d_nope+d_rope)]` then `[bs, nh, d_nope] @ [nh, d_nope, dkv]`. 

The algebraic fusion: `q_nope_latent_h = q_lora @ wUQ_nope_h @ wK_h`. Since `wUQ_nope_h @ wK_h` is a fixed [dq, dkv] = [1536, 512] matrix (per head), we can precompute a fused weight `W_fused = wUQ_nope @ wK` of shape [dq=1536, nh*dkv] = [1536, 65536]. Then `q_nope_latent = q_lora @ W_fused` in a single [bs, 1536] @ [1536, nh*512] GEMM, producing [bs, nh, 512] directly — one GEMM instead of two.

Cache this fused weight on first call (keyed by the weight pointer identity), then use `F.linear` or a raw matmul. This eliminates the `torch.einsum('bhd,hdk->bhk')` call entirely and replaces it with a single large GEMM that cuBLAS can optimize much better. Similarly, fuse `wUQ_rope` to get `q_rope_latent` in one shot.

This is a pure PyTorch-level change with no Triton kernels, no OOM risk, and directly reduces kernel launch count and GEMM overhead in the query path — which is one of the three major phases. The worker should benchmark each phase independently (profile with `torch.cuda.nvtx` if available) to confirm the query projection is actually being improved before fully committing.

---

## Iteration 7

## STATE
Seven experiments, baseline at 2739.78 µs still unbeaten. Experiment #7 (fused weight GEMM) came within 0.5 µs of baseline — essentially noise, confirming the query projection path is not the bottleneck. The score tensor materialization [128, 128, kv_len] remains the clear bottleneck but every attempted kernel fusion has either OOMed or register-spilled. The constraints are hard: dkv=512 is too large for register accumulation; GQA with dkv=512 V OOMs; the baseline's batched GEMM pipeline is already well-optimized by cuBLAS. We've spent 7 iterations finding hard walls.

## RATIONALE
The history reveals one unexplored opportunity: the **score tensor pass itself can be tiled without the dkv=512 problem**. The scores are [bs,nh,kv_len] — each score is a scalar dot product, not a 512-dim vector. A kernel that tiles over `kv_len` and processes multiple batch×head pairs simultaneously, computing scores and doing online softmax + value aggregation in dkv=512 space using **shared memory** (not registers) can work — but only if the tiles are small enough. The H200 has 228KB SMEM per SM. A tile of kv_nope at [tile_kv=16, 512] bf16 = 16KB — fits easily. The accumulator [bs_tile×head_tile, 512] at fp32 = large. 

The unexplored clean path: **two-pass approach with pre-materialized scores**. Pass 1 is the current pipeline (compute scores [bs,nh,kv_len] + softmax). Pass 2 (value aggregation) is just `attn @ kv_nope → M [bs,nh,dkv=512]` + `M @ wV_T → y_head [bs,nh,dv=128]`. These two GEMMs are already done by cuBLAS. The question is: can we reduce the intermediate [bs,nh,kv_len] score tensor traffic? The answer is to use PyTorch's `F.softmax` in place (which cuDNN optimizes) instead of the custom Triton softmax, which makes 3 passes through the data.

## PROPOSAL

**Replace the custom Triton softmax with PyTorch's native `F.softmax`**, and additionally eliminate the `_rotate_half` calls (which create new tensors via `torch.cat`) by using the existing RoPE tables more efficiently. Specifically:

1. Replace `_triton_softmax(scores_flat)` with `torch.softmax(scores_flat, dim=-1)` — PyTorch's softmax dispatches to a fused cuDNN/CUDA kernel that is a single-pass online softmax, reading the row once. The custom Triton implementation makes **three passes**: one for max, one to compute exp and store, one to normalize. The native kernel does it in ~1.5 passes.

2. Replace the `_rotate_half` + `torch.cat` pattern for RoPE (which allocates new tensors) with in-place or slice-based operations: `k_rope_input * cos_k` and the rotate-half can be done without `torch.cat` by operating on the two halves separately using views.

3. Eliminate the custom Triton RoPE kernel for `q_rope` (which launches a separate kernel with 128×128 CTAs each processing 64 elements) — the overhead of launching this tiny kernel exceeds its benefit. Replace with the same elementwise PyTorch ops.

These changes reduce kernel launch overhead and improve the softmax HBM traffic pattern. If PyTorch's softmax is already faster than the 3-pass Triton kernel, this alone could yield a meaningful improvement.

---

## Iteration 8

## STATE
Eight experiments, baseline at 2739.78 µs still unbeaten. The last four experiments have all been within ~200 µs of baseline (2740, 2842, 2917 µs) — these are maturity-noise fluctuations, not real improvements. Every pure PyTorch path is essentially at the cuBLAS optimum already. All Triton attention-fusion attempts have failed due to the dkv=512 dimension constraints (register spill or OOM). We're at 8 iterations with 7 remaining and the score is 3000/2739 ≈ 1.09 vs the roofline ~14.2. There's been no improvement at all.

## RATIONALE
Looking at the compute graph from a fresh angle: the baseline pipeline has five major steps, and the experiments so far have tried to optimize each one. The one step that hasn't been seriously targeted yet is the **KV cache read**: `kv_lora` is stored as [bs, max_seq_len=8192, 576] bf16 = 1.5GB, and every decode step reads [bs, kv_len, 576] from HBM. At kv_len=4096, that's 128×4096×576×2 = 603MB of KV cache reads. This HBM traffic dominates everything else and explains why the baseline is 2739 µs (vs roofline ~211 µs — a 13× gap is consistent with heavy HBM-bound access).

The real fix is to ensure the KV cache reads are **coalesced and sequential** and that the kv_nope slice [bs, kv_len, 512] and k_rope slice [bs, kv_len, 64] are read together as the natural stride [bs, kv_len, 576] — which the baseline already does via `kv_lora[..., :dkv]` and `kv_lora[..., dkv:]` slicing. These slices are not contiguous along the last dim (stride=(576,1) for kv_lora), so cuBLAS GEMMs operating on them may suffer from non-contiguous memory patterns.

The highest-leverage untried move: **make kv_nope_input contiguous** before the GEMMs. `kv_nope_input = kv_lora[..., :dkv]` is a slice along the last dimension with stride 576 (not 512), so it's non-contiguous. Calling `.contiguous()` on it creates a [bs, kv_len, 512] tensor with stride 512, which cuBLAS accesses much more efficiently. The cost is one contiguous copy, but the benefit is that all subsequent reads of kv_nope_input (used 3 times: once for scores_nope, once for M, once for k_rope transpose) become fully coalesced.

## PROPOSAL

Make **kv_nope_input and k_rope_input contiguous** immediately after slicing from kv_lora, before any GEMM operations. The key insight: `kv_lora[..., :512]` and `kv_lora[..., 512:]` are non-contiguous slices with stride 576 on the last dim, which causes strided HBM access patterns in every downstream matmul. A single `.contiguous()` call materializes each as a properly strided [bs, kv_len, 512] / [bs, kv_len, 64] tensor. This pays one copy (603MB + ~75MB ≈ 680MB write) but eliminates strided access in 4+ subsequent GEMMs/matmuls. On H200 with 3.35 TB/s bandwidth, 680MB costs ~200 µs — breakeven if even one matmul is accelerated by 50 µs.

Additionally, try computing `scores_nope` and `scores_rope` in float32 (upcast kv_nope and k_rope to fp32 before matmul) to verify whether the bf16 math precision causes the GEMMs to use a slower bf16 path vs a faster fp32 GEMM on H200. H200 has higher effective throughput for fp16/bf16 tensor cores when inputs are properly aligned — but this is secondary to the contiguity fix.

The worker should implement the contiguity fix alone first and report the result before adding anything else.

---

## Iteration 9

## STATE
Nine experiments total, baseline at 2739.78 µs unbeaten. Nine experiments have failed to improve. The last five attempts have all landed within 200–950 µs of baseline — cuBLAS is already near-optimal for these individual ops. The hard constraints are: dkv=512 blocks flash attention fusion; SDPA GQA OOMs; custom Triton kernels register-spill or crash; PyTorch ops are near-cuBLAS-optimal. The ~13× gap to roofline is real but cannot be closed with any local operation fusion — it reflects the fundamental HBM-bound nature of the problem at these dimensions.

## RATIONALE
After exhausting conventional approaches, I need to reconsider what the roofline is actually measuring. The problem spec says the SOL is ~211 µs at prefill=4096 — this implies ~1.5TB/s effective bandwidth utilization for a nearly ideal implementation. The baseline at 2739 µs is processing the same data at roughly 113 GB/s effective bandwidth — badly underutilizing the H200's 3.35 TB/s peak. The key question is: **what's actually slow?** The KV cache read at [128, 4096, 576] = 603MB alone takes ~180 µs at peak bandwidth. The score tensor [128, 128, 4096] = 134MB adds ~40 µs. The full baseline should be around ~500–600 µs minimum if just bandwidth-bound, yet it takes 2739 µs — **5× slower than memory-bound**. This strongly suggests the bottleneck is **compute-bound GEMMs**, not HBM traffic.

The score computation: `q_nope_latent [128,128,512] @ kv_nope [128,512,4096]` is a batch of 128 GEMMs of shape (128×1×512) @ (128×512×4096) — each GEMM has an M=128, N=4096, K=512 shape, which gives only 128×4096×512×2/1e12 ≈ 0.134 TFLOP. At H200's 989 TFLOPS bf16 tensor core peak, this alone takes only ~136 µs. But this matmul is batched over bs=128, so total is 128× more: 128 × 0.134 TFLOP = 17 TFLOP. At 989 TFLOPS that's only 17 µs — so GEMMs should be very fast.

Something is deeply wrong with my mental model. The baseline is 2739 µs for what should be ~500 µs of memory bandwidth + ~50 µs of compute. The likely culprit: **Python overhead and kernel launch latency** from 15+ separate kernel launches, each with synchronization barriers. With CUDA graphs or `torch.cuda.stream` pipelining, we could potentially eliminate much of this overhead.

## PROPOSAL

Try **CUDA Graph capture** of the hot attention path. The key insight: after `kv_cache` updates (which has side effects and cannot be captured), the remaining computation (RoPE, score matmuls, softmax, value aggregation, projections) is a pure tensor computation with fixed shapes. Capture this as a CUDA graph keyed on `kv_len`, then replay it with a static input buffer.

Concretely: after calling `kv_cache(kv_lora_input)` to update the cache and get `kv_len`, copy `x` and relevant tensors into static GPU buffers, then replay a pre-captured CUDA graph. The graph eliminates all Python-side launch overhead and CPU-GPU synchronization for the ~15 kernel launches. Cache the graph per `kv_len` value (since seq dimension changes).

If CUDA graph capture is not feasible due to the varying `kv_len`, alternatively try `torch.cuda.make_graphed_callables` on just the deterministic portion, or use **CUDA stream overlap** to pipeline the wDQ/wDKV GEMMs with the kv_cache update.

As a simpler fallback if graphs are complex to implement: try computing `wDQ` and `wDKV` projections in parallel using two separate CUDA streams (`stream1` for the Q path, `stream2` for the KV path), then synchronize. This overlaps two independent GEMMs that are currently sequential.

---

## Iteration 10

## STATE
Ten experiments, baseline at 2739.78 µs still unbeaten through the entire run. Every single approach has either crashed, OOMed, or landed within noise of the baseline. The evidence is comprehensive: cuBLAS GEMMs are near-optimal, PyTorch ops are saturating the pipeline, custom Triton kernels fail on dkv=512 constraints, CUDA graph/stream tricks crash, and all algebraic reformulations are equivalent in cost. The roofline gap (~13×) is real but unreachable with any of the explored directions.

## RATIONALE
I need to step back completely and think about what the score board actually reveals. The baseline is at 2739 µs — ~13× above roofline. But the competition has a score of `3000 / geomean_us`. With the baseline at 2739 µs, the score is ~1.09. To reach a meaningful score (say 5), we need ~600 µs. The problem is memory-bound at its core — the KV cache read alone is [128, 4096, 576] = 603MB, which at H200 peak bandwidth takes ~180 µs. The baseline takes 2739 µs — 5× what bandwidth allows. This 5× overhead can only come from serialized, non-overlapping operations where each cuBLAS kernel launches sequentially and the H200 SMs sit idle between launches.

One unexplored avenue: **the entire computation can potentially be mapped to a pure cuBLAS GEMM with `torch.cuda.amp.autocast`** disabled and using **half-precision accumulation**. But more importantly, looking at what the `flash-attn` library actually does for MLA: it uses a **two-pass split-KV** approach where the score computation and the value aggregation are handled in a single fused operation across the entire KV dimension. The `flash_attn_with_kvcache` function in flash-attn v2.6+ supports exactly this. Let me check if it's available.

More practically — the **`vllm` and `SGLang` projects have open-source MLA decode kernels**. The actual production solution for DeepSeek-V3 MLA decode is **not** a from-scratch Triton kernel but rather uses flash-attn's absorption API or a precomputed-KV approach where the KV is materialized into per-head form as a preprocessing step. The key insight: if we precompute `K_heads = kv_nope @ wK^T` [bs, kv_len, nh, d_nope=128] and `V_heads = kv_nope @ wV^T` [bs, kv_len, nh, dv=128] **once per sequence** (caching them), then SDPA on [bs, nh, 1, 128] Q with [bs, nh, kv_len, 128] K/V works perfectly with head_dim=128.

The problem is these precomputed caches are [128, 4096, 128, 128] = 8GB each — too large. BUT: each call only needs the current sequence's KV, and if we precompute them into a fixed-size buffer of [bs, max_seq_len=8192, nh, head_dim=128], that's only 16GB per tensor — still too large.

## PROPOSAL

Try a **fundamentally different algorithmic approach**: precompute and cache the per-head K and V projections in a persistent buffer. Specifically:

Maintain a persistent cache `K_cache [bs, max_seq_len, nh, d_nope]` and `V_cache [bs, max_seq_len, nh, dv]`. On each decode step: (1) compute the new KV entry from `x` and store it in the standard kv_cache; (2) project only the **new token's** `kv_nope` [bs, 1, dkv] through `wK` and `wV` to get `K_new [bs, 1, nh, d_nope]` and `V_new [bs, 1, nh, dv]`, appending to the persistent caches; (3) use `F.scaled_dot_product_attention` with the pre-projected K [bs, nh, kv_len, d_nope] and V [bs, nh, kv_len, dv] using standard head_dim=128. This keeps the SDPA to head_dim=128 (not 512), enabling Flash Attention to work without OOM.

The persistent cache would be [bs=128, max_seq_len=8192, nh=128, d_nope=128] = 128×8192×128×128×2 bytes ≈ 34GB — still too large for per-head materialization. For the benchmark sizes (kv_len≤6144): [128, 6144, 128, 128] = 25GB — still too large.

Given the persistent memory constraints, the only viable path is to compute K and V on-the-fly per batch but without the persistent cache. At kv_len=4096: [128, 128, 4096, 128] = 8GB — too large. But if we **process a subset of heads** at a time (e.g., 16 heads per pass, doing 8 passes), each pass would require [128, 16, 4096, 128] = 1GB per K/V — feasible on H200's 141GB HBM.

Given all the failed explorations, the most pragmatic path forward with 5 iterations remaining is to **try `flash_attn` directly** if available in the environment, as it has a battle-tested MLA path. The worker should check if `flash_attn` is importable and if `flash_attn.flash_attn_with_kvcache` works with the absorbed Q=[bs, 1, nh, dkv] and KV=[bs, kv_len, 1, dkv] (GQA-style). If flash_attn is available, it may have a memory-efficient GQA path that doesn't OOM like PyTorch's SDPA did.

---

## Iteration 11

## STATE
Eleven experiments, zero improvement. Baseline at 2739.78 µs is unbeaten. flash_attn is not available. Every kernel/fusion direction has failed: Triton custom kernels crash on dkv=512, SDPA OOMs, GEMMs are already cuBLAS-optimal, PyTorch ops can't be improved, CUDA streams crash on kv_cache state, and contiguous copies just add traffic. The Triton softmax in the baseline is actually faster than PyTorch's native softmax. With 4 iterations remaining, the situation is dire.

## RATIONALE
A complete reassessment is needed. The key data point from experiment #11: the worker confirmed that k_rope requires materialization because RoPE is applied to it (multiply by cos/sin), and the nope scores come from `q_nope_latent @ kv_nope^T` where kv_nope is the raw kv_cache slice — no materialization. The two-pass computation is already close to optimal for PyTorch.

One thing that hasn't been tried: **the Triton softmax BLOCK_SIZE** is capped at 1024 for kv_len > 1024. At kv_len=4096, the softmax kernel makes 4 passes over each row (4096/1024 = 4 tiles per pass, with 3 passes = 12 total HBM reads per row). The baseline has 16384 softmax rows. This is a significant bottleneck.

A **properly tuned single-pass online softmax** with BLOCK_SIZE = 4096 (the whole row) would read each row exactly twice (one forward pass computing exp and partial sums, one normalization pass). The existing kernel caps BLOCK_SIZE at 1024 because larger values cause "too large for constexpr" — but Triton supports constexpr block sizes up to 8192+. The fix: use `BLOCK_SIZE=4096` for kv_len=4096 and `BLOCK_SIZE=8192` for kv_len=6144 (next power of 2). This turns 3 passes into 2 passes, reducing softmax HBM traffic by 33%.

But actually the current kernel is 3-pass (max, exp+store, normalize). A **proper 2-pass online softmax** computes running max+sum in one pass, then normalizes in a second pass — eliminating the intermediate bf16 store. This reduces total HBM traffic from 3×(read+write) to 2×read+1×write.

## PROPOSAL

**Rewrite the Triton softmax kernel as a proper 2-pass implementation** (one forward pass for max+exp accumulation, one backward normalization pass), with larger BLOCK_SIZE. Key changes:

1. **Increase BLOCK_SIZE**: allow up to 8192 (the whole kv_len row fits in one block). Triton supports this — the issue in the baseline was that it capped at 1024. At BLOCK_SIZE=4096 or 8192, the entire row fits in a single pass.

2. **True 2-pass** instead of 3-pass: Pass 1 scans the row computing online max and sum of exp values, accumulating into registers (no intermediate store). Pass 2 normalizes using the accumulated sum and writes the result. This eliminates the intermediate bf16 materialize-then-reload.

3. **Use more warps**: at BLOCK_SIZE=4096, use 16 or more warps to parallelize the row scan across the 4096 columns, keeping all warp reductions in registers.

4. **Keep output in bf16** (which the baseline already does correctly) to keep the subsequent matmul in bf16.

This is a surgical improvement to the one custom kernel that has been shown to work (the baseline softmax beats PyTorch's native), making it faster by reducing HBM pass count. Worker should verify at multiple kv_len values that the larger BLOCK_SIZE still compiles (Triton has a maximum of 131072 elements per block in recent versions).

---

## Iteration 12

## STATE
Twelve experiments, zero improvement. The baseline at 2739.78 µs is unbeatable by any approach tried so far. Every custom kernel, every fusion, every restructuring has been worse. The experiment reports consistently show the same pattern: the baseline's PyTorch ops (cuBLAS GEMMs + the existing 3-pass Triton softmax) are already near-optimal for this workload on H200. The 13× gap to roofline appears to reflect the inherent sequential nature of the 8–10 dependent kernel launches, not any single kernel's inefficiency.

## RATIONALE
With 3 iterations remaining and zero progress, it's time to make a sobering reassessment. The baseline is already very close to what's achievable with the same algorithmic structure. The fundamental bottleneck is that this is decode-time attention with bs=128 and sq=1 — a highly memory-bound workload with poor arithmetic intensity. The 2739 µs baseline is quite plausible for the actual HBM-bound ops. The roofline of 211 µs likely assumes ideal conditions that can't be achieved without a completely different memory layout for the KV cache itself.

One observation that hasn't been exploited: `scores_nope` and `scores_rope` are computed into two separate intermediate [bs,nh,kv_len] tensors that are then added together. These are both memory writes followed by an immediate read (for the add). A Triton kernel that computes `scores_nope + scores_rope` **in a single fused kernel** — reading each input once, summing, scaling, and writing once — would eliminate one full read+write pass of the [128,128,kv_len] score tensor. At kv_len=4096, this tensor is 128×128×4096×2 = 134MB. Eliminating the second write saves 134MB of HBM writes; the add+scale fused into the matmul saves the read of the second score. This is a real, achievable win that fits entirely within PyTorch's operator model.

## PROPOSAL

Write a **Triton kernel that fuses `score_add_scale`**: takes `scores_nope` [bs, nh, kv_len] and `scores_rope` [bs, nh, kv_len] as inputs, computes `(scores_nope + scores_rope) * scale` and writes the result. More importantly: restructure the computation so `scores_nope` is written to a pre-allocated output buffer, and then `scores_rope` is added into it in-place. This eliminates the temporary allocation and reduces peak memory pressure.

But even more impactful: **fuse the add+scale with the Triton softmax** into a single kernel. Currently there are three separate passes: (1) matmul → scores_nope, (2) matmul → scores_rope, (3) scores_nope + scores_rope * scale → scores, (4) softmax(scores) → attn. Steps 3+4 can be merged: a single Triton kernel takes two input arrays (scores_nope, scores_rope), computes their sum+scale in-place during the softmax max-finding pass, eliminating one HBM write+read of the combined scores tensor. This saves ~268MB of HBM traffic (134MB write of combined scores + 134MB read for softmax).

The fused kernel signature: `fused_add_scale_softmax(out, a, b, scale, n_cols, BLOCK_SIZE)` — computes `softmax((a+b)*scale)` in-place, reading `a` and `b` together and never writing the intermediate sum to HBM.

---

## Iteration 13

## STATE
Thirteen experiments, first improvement at #13: 2631.82 µs (down from 2739.78 µs baseline, ~4% gain). The fused `add+scale+softmax` kernel works by reading `scores_nope` and `scores_rope` directly without materializing their sum to HBM, saving ~268MB of HBM traffic. Per-case data: prefill=4096 improved dramatically (~2025 µs), while prefill=6144 shows high variance (3419 µs ± 617 µs). This variance at kv_len=6144 is a concern — it suggests the kernel is near a boundary (BLOCK_SIZE=1024 still applies since 6144 > 1024's next power of 2). The fused kernel is still 3-pass, so there's more room. With 2 iterations remaining, we must push this direction.

## RATIONALE
The win confirmed the HBM-traffic-reduction hypothesis. The experiment result details show prefill=4096 improved more than prefill=6144 (which has high variance). The current fused kernel still does **3 HBM passes** (max, exp+store, normalize), but now reads 2 arrays instead of 1 per pass. The intermediate bf16 store in pass 2 is still there. The next optimization in the same direction: eliminate the intermediate store entirely by fusing the value aggregation into the softmax, but that's the dkv=512 problem again.

However, there's another fusion that's now unlocked by having a working kernel: **fuse the value-weighted sum into the same kernel**. Since after softmax we immediately compute `M = attn @ kv_nope`, we could continue in the same Triton kernel: after computing attn weights per row, multiply them by `kv_nope` and accumulate into an output buffer of size [bs*nh, dkv]. This would eliminate the entire `attn` [bs,nh,kv_len] tensor write+read (another 134MB each way). But dkv=512 in registers — the problem from exp #3 and #6. 

The key insight that makes this feasible now: instead of accumulating dkv=512 fp32 per thread (2KB registers), **accumulate a slice of dv=128 per head using wV projection inline**. Since wV is [nh, dv, dkv] = per-head, and each thread handles one row (one batch×head pair), load wV_h [dv, dkv] = 512×128×2 = 131KB into shared memory (fits), then for each KV tile, compute `weight * kv_tile @ wV_h^T` — accumulating [dv=128] floats in registers, not [dkv=512]. This was proposed in exp #6 but crashed due to wrong tile sizing. Now we know: use small kv tiles (e.g., 16 tokens), so kv_tile = [16, 512] = 16KB in SMEM.

## PROPOSAL

Extend the current winning `_fused_add_scale_softmax_kernel` to also **perform the value aggregation**, eliminating the `attn @ kv_nope` matmul entirely. The key design change from failed exp #6: **accumulate in dv=128 output space, not dkv=512 latent space**, by loading `wV_h` [dv=128, dkv=512] = 131KB into shared memory at the start of each CTA.

Kernel design for one CTA (one bs×head pair):
1. Load `wV_h` [dv, dkv] into shared memory at kernel start (131KB — fits in H200's 228KB SMEM per SM)
2. First loop pass: compute `(scores_nope + scores_rope) * scale` for max reduction (online softmax pass 1) — same as current
3. Second loop pass: for each KV tile (small, e.g., 16 tokens): load `kv_nope_tile` [tile, dkv] and `scores_nope_tile`+`scores_rope_tile`, compute exp(score - max), accumulate `exp_weight * (kv_nope_tile @ wV_h^T)` into [dv] fp32 accumulator in registers. Also accumulate exp_sum for normalization.
4. After loop: normalize accumulator by exp_sum → output [dv] per head
5. Apply wV projection is already done inside loop — no separate GEMM needed for this head

This eliminates: (1) writing attn [bs,nh,kv_len] to HBM (134MB), (2) reading kv_nope for the value GEMM (603MB already read for scores), (3) writing M [bs,nh,dkv] (128×128×512×2 = 16MB). Total savings: ~268MB + removed one GEMM and one einsum. The accumulator is only [dv=128] fp32 = 512 bytes/thread — very manageable.

The worker should be careful about: (a) wV_h size: [128, 512] × 2 bytes = 131KB — verify this fits in SMEM with the KV tile; (b) tile size: use a small tile (8–32 tokens) so kv_tile fits alongside wV_h; (c) output: the kernel writes [bs×nh, dv] directly, skipping the intermediate M tensor and the value einsum.

---

## Iteration 14

## STATE
Fourteen experiments. Best time: 2631.82 µs (experiment #13 — fused add+scale+softmax). Experiment #14 crashed trying to extend that kernel with value aggregation. The experiment report contains a critical insight in the code: even though the `_fused_softmax_vagg_kernel` was written (accumulating `acc [DKV=512]`), the worker noted "DV=128, DKV=512: 65536/128=512 elements/thread → too many" and fell back to storing M [bs×nh, dkv]. This is the last iteration. The experiment #13 code was *reverted to the winning state* — the custom_kernel still uses the working `_fused_add_scale_softmax`. This is our current best at 2631.82 µs.

## RATIONALE
This is the final iteration. The current best (2631.82 µs) already uses the fused softmax kernel. The experiment #14 worker noted the code reverted to exp#13's winning state, so the current submission already has the fused softmax in place. The key observation from the crash: the fused vagg kernel in exp#14's code actually **already accumulates M[bs×nh, dkv] and stores it** — the wV projection is still external. This means the kernel actually eliminates the `attn [bs,nh,kv_len]` write+read (saving ~268MB) but at the cost of a 2-pass (not 3-pass) softmax structure that crashed for large kv_len due to tile sizing.

The cleanest final move: take the exp#13 winning code exactly as-is (it's already correct and faster), and make **one small additional improvement** that doesn't risk correctness — apply the fused weight projection from exp#7 (which was neutral at 2740 µs from the old baseline). Combined with the fused softmax, the weight fusion might now contribute since the overall path is already optimized. However exp#7 was already neutral against the old baseline, so this is risky.

The safest proposal: **submit the current exp#13 state unchanged** — it's the best known result and adding anything risky with one iteration left could crash and end up with the baseline as submission. However, the rules require a proposal. 

The one thing that's genuinely likely to help without risk: the current `_fused_add_scale_softmax` still produces an `attn` tensor that then feeds `M = torch.matmul(attn, kv_nope_input)`. The `attn` write to HBM costs 134MB. The `M` GEMM reads `attn` again. If we can avoid writing `attn` entirely by keeping the softmax output as the input to the value aggregation — but that's exactly exp#14 which crashed. 

However, exp#14's crash was specifically due to `BLOCK_SIZE` for the kv_len dimension being too large when used for 2D kv tiles. The fix is simple: use a **separate, small, fixed constexpr `KV_BLOCK` for the kv dimension** (e.g., 64 or 128 tokens per tile), completely independent of the softmax BLOCK_SIZE. This is a clean, well-scoped kernel that avoids the crash.

## PROPOSAL

Re-attempt the value aggregation fusion from exp#14, but with a correct, crash-safe design. The root cause of exp#14's crash was using the same `BLOCK_SIZE` (up to 2048) for both the softmax column tiles AND the kv_len tiles in the 2D kv accumulation — creating [2048, 512] tensor blocks that exceeded Triton limits.

The fix: keep the existing `_fused_add_scale_softmax` kernel exactly as-is from exp#13 (don't touch what works), and instead write a **separate, minimal Triton kernel** that fuses ONLY the value aggregation step: `M = attn @ kv_nope`. This kernel takes `attn [bs*nh, kv_len]` (bf16, already computed) and `kv_nope [bs, kv_len, 512]` and outputs `M [bs*nh, 512]` directly, without going through a general batched GEMM. Use a fixed small tile over kv_len (e.g., 64 tokens per tile), with each CTA processing one (batch×head) and accumulating 512 fp32 accumulators (4 per thread with 128 threads — manageable). This is simpler than the exp#14 kernel because it doesn't also do softmax — it's a pure weighted sum kernel, which is much easier to implement correctly.

This eliminates the `attn` tensor write+read (~268MB round-trip) by using the existing fused softmax output directly in a second fused kernel. The two kernels together (fused softmax + fused vagg) would eliminate 3 HBM tensors from the pipeline.

---

## Iteration 15

## STATE
Epoch 1 ends with 15 experiments. Best result: 2631.82 µs (experiment #13, fused add+scale+softmax Triton kernel). The final attempt (#15) confirmed that the cuBLAS batched GEMM for `attn @ kv_nope` cannot be beaten with a naive Triton serial kernel — cuBLAS achieves ~25× better parallelism. The experiment #13 state is the final submission, achieving a ~4% improvement over baseline (2739.78 µs). Score: 3000/2631.82 ≈ 1.14.

## RATIONALE
The run has conclusively mapped the solution space. The only successful optimization was the fused `(scores_nope + scores_rope) * scale → softmax` kernel, which saved ~268MB of HBM intermediate tensor traffic. All GEMMs are already cuBLAS-optimal. Custom Triton kernels for the value aggregation or attention phases are worse than cuBLAS due to insufficient parallelism (16K CTAs vs cuBLAS's thousands of independent GEMM tiles). Epoch 2 should start fresh with this understanding.

**For epoch 2, the highest-value unexplored direction is:** precomputing the wV projection *outside* the attention loop by first computing `V_proj = kv_nope @ wV^T` → [bs, kv_len, nh, dv=128], then using `F.scaled_dot_product_attention` with Q=[bs,nh,1,dkv], K=[bs,nh,kv_len,dkv] (expanded from 1 KV head), and V=V_proj as a proper head_dim=128 tensor. The memory cost is [128, 4096, 128, 128] = 4GB for V_proj — manageable on H200's 141GB. This would allow true Flash Attention to run with head_dim=128 on the V side, eliminating the score tensor entirely. The K side is still head_dim=512, but an `attn_mask` trick (pre-computing rope scores as a bias) combined with standard SDPA on head_dim=128 V might work without OOM.

## PROPOSAL

For the start of epoch 2: **pre-project V before attention** using a contiguous [bs, kv_len, nh, dv=128] tensor, then restructure attention to use head_dim=128 SDPA. Specifically:

1. Compute `V_proj = (kv_nope [bs,kv_len,dkv] @ wV_all [dkv, nh*dv]).view(bs,kv_len,nh,dv)` — a single [128, kv_len, 512] @ [512, nh*128] GEMM producing [bs, kv_len, nh*128]. This is one well-shaped cuBLAS call.
2. Transpose to [bs, nh, kv_len, dv=128] — this is the standard SDPA V layout.
3. For Q and K: use Q=[bs,nh,1,dkv] and K=[bs,nh,kv_len,dkv] with the pre-computed `q_nope_latent` as Q and `kv_nope` as K (both dkv=512), passing `scores_rope` as the `attn_mask` to SDPA. This way V has head_dim=128 (no OOM), and SDPA handles the attention with the rope bias in the mask.
4. At kv_len=4096: V_proj = [128, 4096, 128, 128] bf16 = 4GB — feasible. SDPA with Q=[128,128,1,512], K=[128,128,4096,512] may still OOM for K/V output, but if we can pass V=[128,128,4096,128] to SDPA with the nope scores + rope mask, SDPA can output [128,128,1,128] directly.

