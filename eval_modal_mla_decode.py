"""
Deployable Modal H200 evaluator for the MLA decode kernel task.

Evaluation logic mirrors skydiscover benchmarks/gpu_mode/mla_decode exactly.

Deploy once:
    uv run modal deploy eval_modal_mla_decode.py

Then the agent's run_eval.py calls evaluate_kernel.remote(kernel_code).
"""

import modal

# ── Test / benchmark cases (mirrors reference.py) ─────────────────────────────

TEST_CASES = [
    {"batchsize": 128, "dim": 7168, "dq": 1536, "prefill": 128,  "seed": 9247},
    {"batchsize": 128, "dim": 7168, "dq": 1536, "prefill": 512,  "seed": 2197},
    {"batchsize": 128, "dim": 7168, "dq": 1536, "prefill": 1024, "seed": 9107},
    {"batchsize": 128, "dim": 7168, "dq": 1536, "prefill": 2048, "seed": 5291},
]

BENCHMARK_CASES = [
    {"batchsize": 128, "dim": 7168, "dq": 1536, "prefill": 4096, "seed": 9817},
    {"batchsize": 128, "dim": 7168, "dq": 1536, "prefill": 6144, "seed": 5291},
]

SCORE_SCALE = 3000.0

# Benchmark configuration (mirrors reference.py)
BENCH_USE_CUDA_EVENTS = False
BENCH_REL_ERROR = 0.01
BENCH_WALL_TIMEOUT_NS = None
BENCH_NO_GRAD = True
BENCH_MAX_REPEATS = 100
BENCH_MAX_TIME_NS = 10e9
BENCH_WARMUP_STYLE = "timed_calls"

# ── Modal image ───────────────────────────────────────────────────────────────

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .pip_install("triton")
)

app = modal.App("mla-decode-kernel-eval")


# ── Evaluator function ────────────────────────────────────────────────────────

@app.function(gpu="H200", image=image, timeout=600)
def evaluate_kernel(kernel_code: str, mode: str = "leaderboard") -> str:
    import contextlib
    import copy
    import dataclasses
    import gc
    import importlib.util
    import json as _json
    import math
    import os as _os
    import tempfile
    import time
    import traceback

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ── Reference implementation (inlined from reference.py) ─────────────────

    class RoPE(nn.Module):
        def __init__(self, d_model: int):
            super().__init__()
            self.d_model = d_model
            theta = 10000 ** (-torch.arange(0, d_model // 2, dtype=torch.bfloat16) / (d_model // 2))
            self.register_buffer("theta", theta)

        def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
            seq_len = x.size(-2)
            d_model = x.size(-1)
            assert d_model == self.d_model
            seq_idx = torch.arange(start_pos, start_pos + seq_len, device=x.device)
            idx_theta = torch.einsum("s,d->sd", seq_idx, self.theta)
            idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)
            cos = idx_theta2.cos().to(torch.bfloat16)
            sin = idx_theta2.sin().to(torch.bfloat16)
            return x * cos + self.rotate_half(x) * sin

    class KVCache(nn.Module):
        def __init__(self, kv_cache_shape: tuple, **kwargs) -> None:
            super().__init__(**kwargs)
            self.register_buffer("data", torch.zeros(kv_cache_shape, dtype=torch.bfloat16))
            self.seq_len = 0
            self.zero()

        def zero(self) -> None:
            self.data.zero_()

        def get_data(self) -> torch.Tensor:
            return self.data

        def forward(self, c_kv: torch.Tensor):
            assert self.seq_len + c_kv.size(1) <= self.data.size(1), "KV Cache Exceeded"
            self.data = self.data.to(c_kv.dtype)
            self.data[:, self.seq_len : self.seq_len + c_kv.size(1), :] = c_kv
            self.seq_len += c_kv.size(1)
            return self.data[:, : self.seq_len], self.seq_len

    @dataclasses.dataclass
    class Config:
        batch_size: int
        dim: int
        n_heads: int
        q_lora_rank: int
        kv_lora_rank: int
        qk_nope_head_dim: int
        qk_rope_head_dim: int
        v_head_dim: int
        seq_len: int
        max_seq_len: int
        kv_cache_shape: tuple
        Q_proj_down_weight: torch.Tensor
        Q_proj_up_weight: torch.Tensor
        KV_proj_down_weight: torch.Tensor
        KV_proj_up_weight: torch.Tensor
        wo_weight: torch.Tensor

    class MLA(nn.Module):
        def __init__(self, config: Config):
            super().__init__()
            self.dim = config.dim
            self.n_heads = config.n_heads
            self.q_lora_rank = config.q_lora_rank
            self.kv_lora_rank = config.kv_lora_rank
            self.nope_head_dim = config.qk_nope_head_dim
            self.rope_head_dim = config.qk_rope_head_dim
            self.v_head_dim = config.v_head_dim
            self.Q_proj_down = nn.Linear(self.dim, self.q_lora_rank, dtype=torch.bfloat16, bias=False)
            self.KV_proj_down = nn.Linear(self.dim, self.kv_lora_rank + self.rope_head_dim, dtype=torch.bfloat16, bias=False)
            self.Q_proj_up = nn.Linear(self.q_lora_rank, (self.nope_head_dim + self.rope_head_dim) * self.n_heads, dtype=torch.bfloat16, bias=False)
            self.KV_proj_up = nn.Linear(self.kv_lora_rank, (self.nope_head_dim + self.v_head_dim) * self.n_heads, dtype=torch.bfloat16, bias=False)
            self.q_rope = RoPE(self.rope_head_dim)
            self.k_rope = RoPE(self.rope_head_dim)
            self.wo = nn.Linear(self.v_head_dim * self.n_heads, self.dim, dtype=torch.bfloat16, bias=False)

        def forward(self, x: torch.Tensor, kv_cache: KVCache):
            batch_size, seq_len, _ = x.size()
            q_lora = self.Q_proj_down(x)
            kv_lora = self.KV_proj_down(x)
            kv_lora, kv_len = kv_cache(kv_lora)
            query_pos = kv_len - 1
            q_nope_and_rope = self.Q_proj_up(q_lora).view(
                batch_size, seq_len, self.n_heads, self.nope_head_dim + self.rope_head_dim
            )
            q_nope, q_rope = torch.split(q_nope_and_rope, [self.nope_head_dim, self.rope_head_dim], dim=-1)
            kv_nope, k_rope = torch.split(kv_lora, [self.kv_lora_rank, self.rope_head_dim], dim=-1)
            kv_nope = self.KV_proj_up(kv_nope).view(
                batch_size, kv_len, self.n_heads, self.nope_head_dim + self.v_head_dim
            )
            k_nope, v = torch.split(kv_nope, [self.nope_head_dim, self.v_head_dim], dim=-1)
            q_rope = q_rope.permute(0, 2, 1, 3)
            q_rope = self.q_rope(q_rope, start_pos=query_pos)
            q_nope = q_nope.permute(0, 2, 1, 3)
            q = torch.concat([q_nope, q_rope], dim=-1)
            k_rope = k_rope[:, None, :, :]
            k_rope = self.k_rope(k_rope).expand(-1, self.n_heads, -1, -1)
            k_nope = k_nope.permute(0, 2, 1, 3)
            k = torch.concat([k_nope, k_rope], dim=-1)
            v = v.permute(0, 2, 1, 3)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.rope_head_dim + self.nope_head_dim)
            attn = F.softmax(scores, dim=-1).to(torch.bfloat16)
            y = torch.matmul(attn, v).view(batch_size, 1, -1)
            y = self.wo(y)
            return y, kv_cache.get_data()

    def generate_input(batchsize, dim, dq, prefill, seed):
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
        Q_proj_down_weight = torch.randn((dq, dim), dtype=torch.bfloat16, generator=gen, device="cuda") / math.sqrt(dim)
        KV_proj_down_weight = torch.randn((512 + 64, dim), dtype=torch.bfloat16, generator=gen, device="cuda") / math.sqrt(dim)
        Q_proj_up_weight = torch.randn(((128 + 64) * 128, dq), dtype=torch.bfloat16, generator=gen, device="cuda") / math.sqrt(dq)
        KV_proj_up_weight = torch.randn(((128 + 128) * 128, 512), dtype=torch.bfloat16, generator=gen, device="cuda") / math.sqrt(512)
        wo_weight = torch.randn((dim, 128 * 128), dtype=torch.bfloat16, generator=gen, device="cuda") / math.sqrt(128 * 128)
        config = Config(
            batch_size=batchsize, dim=dim, q_lora_rank=dq, n_heads=128,
            kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64,
            v_head_dim=128, seq_len=1, max_seq_len=8192,
            kv_cache_shape=(batchsize, 8192, 512 + 64),
            Q_proj_down_weight=Q_proj_down_weight,
            Q_proj_up_weight=Q_proj_up_weight,
            KV_proj_down_weight=KV_proj_down_weight,
            KV_proj_up_weight=KV_proj_up_weight,
            wo_weight=wo_weight,
        )
        x = torch.randn((config.batch_size, 1, config.dim), dtype=torch.bfloat16, generator=gen, device="cuda")
        kv_cache = KVCache((config.batch_size, config.max_seq_len, config.kv_lora_rank + config.qk_rope_head_dim)).to("cuda")
        pre_filled_cache = torch.randn(
            (config.batch_size, prefill, config.kv_lora_rank + config.qk_rope_head_dim),
            dtype=torch.bfloat16, generator=gen, device="cuda",
        )
        kv_cache(pre_filled_cache)
        return config, x, kv_cache

    def ref_kernel(data):
        config, x, kv_cache = data
        model = MLA(config).to("cuda")
        model.Q_proj_down.weight = nn.Parameter(config.Q_proj_down_weight)
        model.Q_proj_up.weight = nn.Parameter(config.Q_proj_up_weight)
        model.KV_proj_down.weight = nn.Parameter(config.KV_proj_down_weight)
        model.KV_proj_up.weight = nn.Parameter(config.KV_proj_up_weight)
        model.wo.weight = nn.Parameter(config.wo_weight)
        output, kv_data = model(x, kv_cache)
        return output, kv_data

    @torch.no_grad()
    def _verbose_allclose(received, expected, rtol=1e-5, atol=1e-8, max_print=5):
        if received.shape != expected.shape:
            return False, [f"SIZE MISMATCH. received={received.shape}, expected={expected.shape}"]
        diff = torch.abs(received.to(torch.float32) - expected.to(torch.float32))
        tolerance = atol + rtol * torch.abs(expected.to(torch.float32))
        tol_mismatched = diff > tolerance
        nan_mismatched = torch.logical_xor(torch.isnan(received), torch.isnan(expected))
        posinf_mismatched = torch.logical_xor(torch.isposinf(received), torch.isposinf(expected))
        neginf_mismatched = torch.logical_xor(torch.isneginf(received), torch.isneginf(expected))
        mismatched = torch.logical_or(
            torch.logical_or(tol_mismatched, nan_mismatched),
            torch.logical_or(posinf_mismatched, neginf_mismatched),
        )
        mismatched_indices = torch.nonzero(mismatched)
        num_mismatched = mismatched.count_nonzero().item()
        if num_mismatched >= 1:
            details = [f"Number of mismatched elements: {num_mismatched}"]
            for index in mismatched_indices[:max_print]:
                i = tuple(index.tolist())
                details.append(f"ERROR at {i}: received={received[i]}, expected={expected[i]}")
            if num_mismatched > max_print:
                details.append(f"... and {num_mismatched - max_print} more")
            return False, details
        return True, [f"Maximum error: {torch.max(diff)}"]

    def check_implementation(data, submission_output, rtol=2e-2, atol=8e-3):
        output_mla, output_kv = submission_output
        output_mla_cpu = output_mla.cpu()
        output_kv_cpu = output_kv.cpu()
        del output_mla, output_kv
        gc.collect()
        torch.cuda.empty_cache()
        config, x, kv_cache = data
        with torch.no_grad():
            expected_mla, expected_kv = ref_kernel((config, x, kv_cache))
        expected_mla_cpu = expected_mla.cpu()
        expected_kv_cpu = expected_kv.cpu()
        del expected_mla, expected_kv
        gc.collect()
        torch.cuda.empty_cache()
        good_mla, reasons_mla = _verbose_allclose(output_mla_cpu, expected_mla_cpu, rtol=rtol, atol=atol)
        good_kv, reasons_kv = _verbose_allclose(output_kv_cpu, expected_kv_cpu, rtol=rtol, atol=atol)
        if not good_mla:
            return False, "MLA output mismatch: " + " ".join(reasons_mla)
        if not good_kv:
            return False, "KV cache mismatch: " + " ".join(reasons_kv)
        return True, "Match"

    # ── Shared eval helpers ───────────────────────────────────────────────────

    def _clone(data):
        if isinstance(data, tuple):
            return tuple(_clone(x) for x in data)
        if isinstance(data, list):
            return [_clone(x) for x in data]
        if isinstance(data, dict):
            return {k: _clone(v) for k, v in data.items()}
        if isinstance(data, torch.Tensor):
            return data.clone()
        if dataclasses.is_dataclass(data) and not isinstance(data, type):
            fields = {f.name: _clone(getattr(data, f.name)) for f in dataclasses.fields(data)}
            return type(data)(**fields)
        if isinstance(data, torch.nn.Module):
            cloned = copy.deepcopy(data)
            if hasattr(data, "seq_len"):
                cloned.seq_len = data.seq_len
            return cloned
        return data

    def _stats(durations):
        n = len(durations)
        avg = sum(durations) / n
        if n > 1:
            var = sum((x - avg) ** 2 for x in durations) / (n - 1)
            std = math.sqrt(var)
            err = std / math.sqrt(n)
        else:
            std, err = 0.0, 0.0
        return {"runs": n, "mean": avg, "std": std, "err": err}

    def _warmup(kernel_fn, bench_args):
        # Run for 200 ms to trigger Triton compilation and warm up caches
        data = generate_input(**bench_args)
        start = time.perf_counter()
        while time.perf_counter() - start < 0.2:
            kernel_fn(data)
            torch.cuda.synchronize()

    def _bench_single(kernel_fn, bench_args, max_time_ns=None):
        if max_time_ns is None:
            max_time_ns = BENCH_MAX_TIME_NS

        data = generate_input(**bench_args)
        data_copy = _clone(data)

        ctx = torch.no_grad() if BENCH_NO_GRAD else contextlib.nullcontext()
        with ctx:
            output = kernel_fn(data)
            torch.cuda.synchronize()
            passed, msg = check_implementation(data_copy, output)
        if not passed:
            return None, f"Benchmark correctness: {msg}"
        del output

        durations_ns = []
        bm_start = time.perf_counter_ns()

        with ctx:
            for i in range(BENCH_MAX_REPEATS):
                torch.cuda.synchronize()
                t0 = time.perf_counter_ns()
                output = kernel_fn(data)
                torch.cuda.synchronize()
                duration_ns = time.perf_counter_ns() - t0
                del output
                durations_ns.append(duration_ns)

                if i > 1:
                    st = _stats(durations_ns)
                    if st["mean"] > 0 and st["err"] / st["mean"] < BENCH_REL_ERROR:
                        break
                    if st["mean"] * st["runs"] > max_time_ns:
                        break
                    if BENCH_WALL_TIMEOUT_NS is not None and (time.perf_counter_ns() - bm_start) > BENCH_WALL_TIMEOUT_NS:
                        break

        return _stats(durations_ns), None

    # ── GPU info ──────────────────────────────────────────────────────────────

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    torch_ver = torch.__version__

    # ── Load submission ───────────────────────────────────────────────────────

    tmp_dir = tempfile.mkdtemp(prefix="submission_")
    tmp_path = _os.path.join(tmp_dir, "submission.py")

    # Write reference classes to tmp_dir so `from reference import ...` works
    ref_path = _os.path.join(tmp_dir, "reference.py")
    import sys as _sys
    _sys.path.insert(0, tmp_dir)

    # Inline the reference module so submissions can `from reference import KVCache, Config`
    ref_src = '''
import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

class RoPE(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        theta = 10000 ** (-torch.arange(0, d_model // 2, dtype=torch.bfloat16) / (d_model // 2))
        self.register_buffer("theta", theta)
    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    def forward(self, x, start_pos=0):
        seq_len = x.size(-2)
        d_model = x.size(-1)
        seq_idx = torch.arange(start_pos, start_pos + seq_len, device=x.device)
        idx_theta = torch.einsum("s,d->sd", seq_idx, self.theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)
        cos = idx_theta2.cos().to(torch.bfloat16)
        sin = idx_theta2.sin().to(torch.bfloat16)
        return x * cos + self.rotate_half(x) * sin

class KVCache(nn.Module):
    def __init__(self, kv_cache_shape, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("data", torch.zeros(kv_cache_shape, dtype=torch.bfloat16))
        self.seq_len = 0
        self.zero()
    def zero(self):
        self.data.zero_()
    def get_data(self):
        return self.data
    def forward(self, c_kv):
        assert self.seq_len + c_kv.size(1) <= self.data.size(1), "KV Cache Exceeded"
        self.data = self.data.to(c_kv.dtype)
        self.data[:, self.seq_len:self.seq_len + c_kv.size(1), :] = c_kv
        self.seq_len += c_kv.size(1)
        return self.data[:, :self.seq_len], self.seq_len

@dataclass
class Config:
    batch_size: int
    dim: int
    n_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    seq_len: int
    max_seq_len: int
    kv_cache_shape: tuple
    Q_proj_down_weight: object
    Q_proj_up_weight: object
    KV_proj_down_weight: object
    KV_proj_up_weight: object
    wo_weight: object
'''
    with open(ref_path, "w") as f:
        f.write(ref_src)

    with open(tmp_path, "w") as f:
        f.write(kernel_code)

    try:
        spec = importlib.util.spec_from_file_location("submission", tmp_path)
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["submission"] = mod
        spec.loader.exec_module(mod)
        custom_kernel = mod.custom_kernel
    except Exception:
        return _json.dumps({
            "success": False,
            "error": f"Failed to load submission:\n{traceback.format_exc()}",
            "tests_passed": 0,
            "tests_total": len(TEST_CASES),
            "test_details": [],
            "gpu_name": gpu_name,
            "torch_version": torch_ver,
            "platform": "modal-h200",
            "failure_stage": "import",
        })

    # ── Correctness tests ─────────────────────────────────────────────────────

    test_details = []
    tests_passed = 0
    for tc in TEST_CASES:
        try:
            data = generate_input(**tc)
            data_copy = _clone(data)
            torch.cuda.synchronize()
            output = custom_kernel(data)
            torch.cuda.synchronize()
            passed, msg = check_implementation(data_copy, output)
            test_details.append({
                "batchsize": tc["batchsize"], "prefill": tc["prefill"], "seed": tc["seed"],
                "passed": passed,
                "error": "" if passed else msg,
            })
            if passed:
                tests_passed += 1
        except Exception:
            test_details.append({
                "batchsize": tc["batchsize"], "prefill": tc["prefill"], "seed": tc["seed"],
                "passed": False,
                "error": traceback.format_exc()[:800],
            })

    if tests_passed < len(TEST_CASES):
        return _json.dumps({
            "success": False,
            "tests_passed": tests_passed,
            "tests_total": len(TEST_CASES),
            "test_details": test_details,
            "error": "Correctness check failed — see test_details",
            "gpu_name": gpu_name,
            "torch_version": torch_ver,
            "platform": "modal-h200",
            "failure_stage": "correctness",
        })

    if mode == "test":
        return _json.dumps({
            "success": True,
            "tests_passed": tests_passed,
            "tests_total": len(TEST_CASES),
            "test_details": test_details,
            "gpu_name": gpu_name,
            "torch_version": torch_ver,
            "platform": "modal-h200",
        })

    # ── Warmup ────────────────────────────────────────────────────────────────

    gc.collect()
    torch.cuda.empty_cache()
    _warmup(custom_kernel, BENCHMARK_CASES[0])

    # ── Benchmarks ────────────────────────────────────────────────────────────

    benchmark_details = []
    bench_means_ns = []

    for bench_args in BENCHMARK_CASES:
        st, err = _bench_single(custom_kernel, bench_args)
        if err:
            return _json.dumps({
                "success": False,
                "tests_passed": tests_passed,
                "tests_total": len(TEST_CASES),
                "test_details": test_details,
                "error": err,
                "gpu_name": gpu_name,
                "torch_version": torch_ver,
                "platform": "modal-h200",
                "failure_stage": "benchmark",
            })

        mean_us = st["mean"] / 1e3
        err_us = st["err"] / 1e3
        benchmark_details.append({
            "batchsize": bench_args["batchsize"],
            "prefill": bench_args["prefill"],
            "seed": bench_args["seed"],
            "mean_us": round(mean_us, 3),
            "err_us": round(err_us, 3),
            "runs": st["runs"],
        })
        bench_means_ns.append(st["mean"])

    means_s = [ns / 1e9 for ns in bench_means_ns]
    geomean_s = math.pow(math.prod(means_s), 1.0 / len(means_s))
    geomean_us = geomean_s * 1e6
    score = SCORE_SCALE / geomean_us

    return _json.dumps({
        "success": True,
        "tests_passed": tests_passed,
        "tests_total": len(TEST_CASES),
        "test_details": test_details,
        "benchmark": {
            "geomean_us": round(geomean_us, 3),
            "score": round(score, 3),
        },
        "benchmark_details": benchmark_details,
        "gpu_name": gpu_name,
        "torch_version": torch_ver,
        "platform": "modal-h200",
    })
