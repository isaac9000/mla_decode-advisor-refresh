"""
Deployable Modal H100 evaluator for the float16 vector addition kernel task.

Evaluation logic mirrors skydiscover benchmarks/gpu_mode exactly.

Deploy once:
    uv run modal deploy eval_modal_vectoradd.py

Then the agent's run_eval.py calls evaluate_kernel.remote(kernel_code).
"""

import modal

# ── Reference implementation (mirrors skydiscover vecadd/reference.py) ────────

TEST_CASES = [
    {"N": 256,  "seed": 42},
    {"N": 512,  "seed": 123},
    {"N": 1024, "seed": 456},
    {"N": 2048, "seed": 789},
]

BENCHMARK_CASES = [
    {"N": 1024, "seed": 1001},
    {"N": 2048, "seed": 1002},
    {"N": 4096, "seed": 1003},
    {"N": 8192, "seed": 1004},
]

SCORE_SCALE = 3000.0
BENCH_USE_CUDA_EVENTS = True
BENCH_REL_ERROR = 0.001       # stop when stderr/mean < 0.1%
BENCH_MAX_REPEATS = 100
BENCH_MAX_TIME_NS = 10e9      # 10 s per case
BENCH_WALL_TIMEOUT_NS = 120e9 # 120 s wall time
BENCH_NO_GRAD = False
BENCH_WARMUP_STYLE = "tiny_benchmark"  # 10 ms warmup run

# ── Modal image ───────────────────────────────────────────────────────────────

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .pip_install("triton")
)

app = modal.App("vectoradd-kernel-eval")


# ── Evaluator function ────────────────────────────────────────────────────────

@app.function(gpu="H100", image=image, timeout=300)
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

    # ── Reference helpers ────────────────────────────────────────────────────

    def ref_kernel(data):
        a, b = data
        return a + b

    def generate_input(N, seed):
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
        a = torch.randn(N, N, device="cuda", dtype=torch.float16, generator=gen)
        b = torch.randn(N, N, device="cuda", dtype=torch.float16, generator=gen)
        return (a, b)

    def check_implementation(data, output, rtol=1e-3, atol=1e-3):
        ref_out = ref_kernel(data)
        if output.shape != ref_out.shape:
            return False, f"Shape mismatch: expected {ref_out.shape}, got {output.shape}"
        if output.dtype != torch.float16:
            return False, f"Dtype mismatch: expected float16, got {output.dtype}"
        if torch.allclose(output, ref_out, rtol=rtol, atol=atol):
            return True, "Match"
        diff = torch.abs(output.float() - ref_out.float())
        return False, f"Output mismatch: max_diff={diff.max().item():.6f}"

    # ── Shared eval helpers (mirrors shared_eval.py) ─────────────────────────

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
            return copy.deepcopy(data)
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

                if BENCH_USE_CUDA_EVENTS:
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    output = kernel_fn(data)
                    e.record()
                    torch.cuda.synchronize()
                    duration_ns = s.elapsed_time(e) * 1e6  # ms -> ns
                else:
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
                    if (time.perf_counter_ns() - bm_start) > BENCH_WALL_TIMEOUT_NS:
                        break

        return _stats(durations_ns), None

    def _warmup(kernel_fn, bench_args):
        if BENCH_WARMUP_STYLE == "timed_calls":
            data = generate_input(**bench_args)
            start = time.perf_counter()
            while time.perf_counter() - start < 0.2:
                kernel_fn(data)
                torch.cuda.synchronize()
        else:
            _bench_single(kernel_fn, bench_args, max_time_ns=10e7)  # 10 ms

    # ── Load submission ───────────────────────────────────────────────────────

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    torch_ver = torch.__version__

    tmp_dir = tempfile.mkdtemp(prefix="submission_")
    tmp_path = _os.path.join(tmp_dir, "submission.py")
    with open(tmp_path, "w") as f:
        f.write(kernel_code)

    try:
        spec = importlib.util.spec_from_file_location("submission", tmp_path)
        mod = importlib.util.module_from_spec(spec)
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
            "platform": "modal-h100",
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
                "N": tc["N"], "seed": tc["seed"],
                "passed": passed,
                "error": "" if passed else msg,
            })
            if passed:
                tests_passed += 1
        except Exception:
            test_details.append({
                "N": tc["N"], "seed": tc["seed"],
                "passed": False,
                "error": traceback.format_exc()[:600],
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
            "platform": "modal-h100",
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
            "platform": "modal-h100",
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
                "platform": "modal-h100",
                "failure_stage": "benchmark",
            })

        mean_us = st["mean"] / 1e3
        err_us = st["err"] / 1e3
        benchmark_details.append({
            "N": bench_args["N"],
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
        "platform": "modal-h100",
    })
