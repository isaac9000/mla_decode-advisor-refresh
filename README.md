# GPU Kernel Autoresearch

An advisor-worker agent pair that iteratively optimizes CUDA kernels on NVIDIA H100. Each iteration the **advisor** reviews experiment history and proposes a strategic direction; the **worker** implements it, evaluates on an H100 via Modal, and logs the result.

Two tasks are supported:
- **grayscale/** — RGB-to-grayscale conversion (float32)
- **vectoradd/** — float16 element-wise vector addition

## Task

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

Deploy the H100 evaluator (once, before any agent runs):

```bash
uv run modal deploy eval_modal_vectoradd.py
```

## Running the agent

### VectorAdd (float16)

```bash
uv run vectoradd/agent.py --iterations 20
```

Start from the provided starting point:

```bash
uv run vectoradd/agent.py --baseline vectoradd/starting_point.py --iterations 20
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

### Grayscale (float32)

```bash
uv run grayscale/agent.py --iterations 20
```

Start from a specific baseline file:

```bash
uv run grayscale/agent.py --baseline grayscale/submission.py --iterations 20
```

Use different models for advisor and worker:

```bash
uv run grayscale/agent.py --advisor-model claude-opus-4-8 --worker-model claude-sonnet-4-6 --iterations 20
```

## Structure

```
eval_modal_grayscale.py   — deployable Modal H100 evaluator (grayscale)
eval_modal_vectoradd.py   — deployable Modal H100 evaluator (vectoradd)
grayscale/
├── agent.py              — advisor-worker agentic loop
├── advisor_prompt.md     — advisor system prompt: strategy, comparison discipline
├── worker_prompt.md      — worker system prompt: mandatory sequence, rules
├── submission.py         — the kernel file the worker edits each iteration
├── run_eval.py           — submits submission.py to the deployed Modal evaluator
├── tools.py              — log_experiment and get_experiment_history tools
└── runs/                 — one directory per run: history, TSV log, plots, best submission
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

Each run directory contains:
- `experiment_history.md` — full log of every attempt with code and result
- `results.tsv` — tab-separated summary for plotting
- `progress.png` — latency scatter plot updated each experiment; shows keep/discard/crash points, best-time step line, and cumulative LLM call count
- `iterations.png` — best latency per advisor iteration
- `best_submission.py` — snapshot of the fastest kernel found so far
- `proposals.md` — advisor proposals for every iteration
- `snapshot_iter{N}.py` — per-iteration snapshot of submission.py before the worker edits it

## LLM Call Counter

The agent tracks how many times the LLM is invoked across both the advisor and worker agents (each tool-calling turn and each plain response counts as one call). This is reported:

- **Per-iteration** in the console: `[advisor]` and `[worker]` call counts accumulated into a running total
- **At each checkpoint** (every `--checkpoint-every` iterations): `LLM calls (total): T`
- **In the final report**: `LLM calls (total): T`
- **On `progress.png`**: displayed as a badge in the bottom-right corner of every plot, updated live as experiments are logged
