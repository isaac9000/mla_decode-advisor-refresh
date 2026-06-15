# GPU Kernel Autoresearch

Two advisor-worker agent pairs that iteratively optimize GPU kernels on Modal cloud GPUs. Each iteration the **advisor** reviews experiment history and proposes a strategic direction; the **worker** implements it, evaluates on a cloud GPU via Modal, and logs the result.

## Problems

| Problem | GPU | Task |
|---|---|---|
| [vectoradd](#vectoradd) | H100 | Float16 element-wise matrix addition |
| [mla_decode](#mla_decode) | H200 | DeepSeek-V3 Multi-Head Latent Attention decode |

---

## Setup

```bash
uv sync
```

Create a `.env` file in the repo root:

```
ANTHROPIC_API_KEY=...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
AUTORESEARCH_MODEL=claude-sonnet-4-6   # optional, this is the default
```

---

## vectoradd

An advisor-worker agent pair that iteratively optimizes a CUDA kernel for float16 vector addition on NVIDIA H100.

### Task

Add two `(N, N)` float16 matrices element-wise:

```
C = A + B
```

`custom_kernel` receives a `(A, B)` tuple and returns a new tensor:

| Argument | Shape | Dtype |
|---|---|---|
| A | `N × N` | `float16` |
| B | `N × N` | `float16` |
| output | `N × N` | `float16` |

**Correctness test shapes** (must pass before benchmarking):

| N |
|---|
| 256 |
| 512 |
| 1024 |
| 2048 |

**Benchmark shapes:**

| N | Elements |
|---|---|
| 1024 | 1024 × 1024 |
| 2048 | 2048 × 2048 |
| 4096 | 4096 × 4096 |
| 8192 | 8192 × 8192 |

Ranked by geometric mean latency across all four benchmark shapes (lower is better). Score = 3000 / geomean_us.

### Deploy the evaluator

```bash
uv run modal deploy eval_modal_vectoradd.py
```

### Running the agent

```bash
uv run vectoradd/agent.py --iterations 20
```

Start from the provided starting point:

```bash
uv run vectoradd/agent.py --baseline vectoradd/starting_point.py --iterations 20
```

Use different models for advisor and worker:

```bash
uv run vectoradd/agent.py --advisor-model claude-opus-4-8 --worker-model claude-sonnet-4-6 --iterations 20
```

Or use the provided script (checks for H100 then launches in tmux):

```bash
./run_agent.sh
```

Evaluate a kernel file without running the agent:

```bash
cd vectoradd
python run_eval.py submission.py -o results.json
python run_eval.py submission.py -o results.json --mode test   # correctness only
```

### Structure

```
eval_modal_vectoradd.py   — deployable Modal H100 evaluator
run_agent.sh              — H100 check + tmux agent launcher
vectoradd/
├── agent.py              — advisor-worker agentic loop
├── advisor_prompt.md     — advisor system prompt: strategy, comparison discipline
├── worker_prompt.md      — worker system prompt: mandatory sequence, rules
├── submission.py         — the kernel file the worker edits each iteration
├── starting_point.py     — original Triton baseline
├── run_eval.py           — submits submission.py to the deployed Modal evaluator
├── tools.py              — log_experiment and get_experiment_history tools
└── runs/                 — one directory per run: history, TSV log, plots, best submission
```

---

## mla_decode

An advisor-worker agent pair that iteratively optimizes a CUDA kernel for the decode step of DeepSeek-V3's Multi-Head Latent Attention on NVIDIA H200.

### Task

Run one decode step of MLA with a pre-filled KV cache:

```
attn_output, kv_cache_data = custom_kernel((config, x, kv_cache))
```

`custom_kernel` receives a `(Config, x, KVCache)` tuple and returns a `(attn_output, kv_cache.data)` pair:

| Argument | Shape | Dtype |
|---|---|---|
| x | `[bs, sq=1, dim]` | `bfloat16` |
| kv_cache.data | `[bs, max_seq_len, kv_lora_rank + qk_rope_head_dim]` | `bfloat16` |
| attn_output | `[bs, sq=1, dim]` | `bfloat16` |
| kv_cache.data (out) | `[bs, max_seq_len, kv_lora_rank + qk_rope_head_dim]` | `bfloat16` |

**Model dimensions (DeepSeek-V3):**

| Parameter | Value |
|---|---|
| bs | 128 |
| dim | 7168 |
| n_heads | 128 |
| q_lora_rank | 1536 |
| kv_lora_rank | 512 |
| qk_nope_head_dim | 128 |
| qk_rope_head_dim | 64 |
| v_head_dim | 128 |

**Correctness test shapes** (must pass before benchmarking):

| prefill |
|---|
| 128 |
| 512 |
| 1024 |
| 2048 |

**Benchmark shapes:**

| prefill | Roofline SOL (µs) |
|---|---|
| 4096 | ~210.75 |
| 6144 | ~280.87 |

Ranked by geometric mean latency across both benchmark shapes (lower is better). Score = 3000 / geomean_us.

### Deploy the evaluator

```bash
uv run modal deploy eval_modal_mla_decode.py
```

### Running the agent

```bash
uv run mla_decode/agent.py --epoch-sizes 15 15
```

Start from the provided starting point:

```bash
uv run mla_decode/agent.py --baseline mla_decode/starting_point.py --epoch-sizes 15 15
```

Use different models for advisor and worker:

```bash
uv run mla_decode/agent.py --advisor-model claude-opus-4-8 --worker-model claude-sonnet-4-6 --epoch-sizes 15 15
```

Evaluate a kernel file without running the agent:

```bash
cd mla_decode
python run_eval.py submission.py -o results.json
python run_eval.py submission.py -o results.json --mode test   # correctness only
```

### Structure

```
eval_modal_mla_decode.py  — deployable Modal H200 evaluator
mla_decode/
├── agent.py              — advisor-worker agentic loop
├── advisor_prompt.md     — advisor system prompt: strategy, comparison discipline
├── worker_prompt.md      — worker system prompt: mandatory sequence, rules
├── reference.py          — reference implementation: RoPE, KVCache, Config, MLA, generate_input
├── submission.py         — the kernel file the worker edits each iteration
├── starting_point.py     — Triton softmax + RoPE baseline
├── run_eval.py           — submits submission.py to the deployed Modal evaluator
├── tools.py              — log_experiment and get_experiment_history tools
└── runs/                 — one directory per run: history, TSV log, plots, best submission
```

---

## Run directories

Each run directory (under `mla_decode/runs/`) contains one subdirectory per epoch (timestamp-named). Each epoch directory contains:
- `experiment_history.md` — full log of every attempt with code and result (deleted after epoch commit)
- `results.tsv` — tab-separated summary for plotting (deleted after epoch commit)
- `progress.png` — latency scatter plot updated each experiment; shows keep/discard/crash points, best-time step line, and cumulative LLM call count
- `iterations.png` — best latency per advisor iteration
- `best_submission.py` — snapshot of the fastest kernel found so far (preserved as next epoch's baseline)
- `proposals.md` — advisor proposals for every iteration
- `snapshot_iter{N}.py` — per-iteration snapshot of submission.py before the worker edits it

At the end of each epoch the directory is git-committed and artifacts are cleared so the next epoch's agents start blind.

## LLM Call Counter

The agent tracks how many times the LLM is invoked across both the advisor and worker agents (each tool-calling turn and each plain response counts as one call). This is reported:

- **Per-iteration** in the console: `[advisor]` and `[worker]` call counts accumulated into a running total
- **At each checkpoint** (every `--checkpoint-every` iterations): `LLM calls (total): T`
- **In the final report**: `LLM calls (total): T`
- **On `progress.png`**: displayed as a badge in the bottom-right corner of every plot, updated live as experiments are logged
