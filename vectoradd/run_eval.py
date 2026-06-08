#!/usr/bin/env python3
"""
CLI wrapper that submits a float16 vector add kernel to the deployed Modal H100 evaluator
and writes results.json in markdown format the agent can parse.

Deploy the evaluator once before running:
    uv run modal deploy eval_modal_vectoradd.py

Usage:
    python run_eval.py submission.py -o results.json
    python run_eval.py submission.py -o results.json --mode test   # correctness only
"""

import argparse
import json
import sys
import threading

import modal

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


def format_results_markdown(res: dict, mode: str = "leaderboard") -> str:
    gpu = res.get("gpu_name", "NVIDIA H100")
    torch_ver = res.get("torch_version", "unknown")
    plat = res.get("platform", "unknown")

    if res["success"]:
        status_line = "**H100 on Modal ✅ success**"
    else:
        status_line = "**H100 on Modal ❌ failure**"

    lines = [status_line]

    if res["success"]:
        lines.append("> ✅ Testing successful")
        if mode == "leaderboard":
            lines.append("> ✅ Benchmarking successful")
    elif res.get("tests_passed", 0) == res.get("tests_total", 1):
        lines.append("> ✅ Testing successful")
        lines.append("> ❌ Benchmarking failed")
    else:
        lines.append("> ❌ Testing failed")

    lines += [
        "",
        "Running on:",
        f"* GPU: `{gpu}`",
        f"* Runtime: `CUDA`",
        f"* Platform: `{plat}`",
        f"* Torch: `{torch_ver}`",
        "",
    ]

    passed = res.get("tests_passed", 0)
    total = res.get("tests_total", 0)
    lines.append(f"## {'✅' if passed == total else '❌'} Passed {passed}/{total} tests:")
    lines.append("```")
    for td in res.get("test_details", []):
        icon = "✅" if td["passed"] else "❌"
        seed_info = f" seed={td['seed']}" if "seed" in td else ""
        n_val = td.get("N", td.get("size", "?"))
        lines.append(f"{icon} N={n_val}{seed_info}")
        if td.get("error"):
            lines.append(f"   ERROR: {td['error']}")
    lines.append("```")

    if res.get("error") and not res["success"]:
        lines += ["", "## Error:", "```", res["error"], "```"]

    bm = res.get("benchmark")
    if bm and mode == "leaderboard":
        geomean = bm["geomean_us"]
        score = bm.get("score", "")
        lines += ["", "## Benchmarks:", "```", f"Geometric mean: ⏱ {geomean} µs", ""]
        if score:
            lines.append(f"Score: {score}")
            lines.append("")
        for bd in res.get("benchmark_details", []):
            n_val = bd.get("N", bd.get("size", "?"))
            seed_str = f" seed={bd['seed']}" if "seed" in bd else ""
            lines.append(
                f"  N={n_val}{seed_str}: ⏱ {bd['mean_us']} ± {bd['err_us']} µs"
                f"  (runs={bd.get('runs', '?')})"
            )
        lines.append("```")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a float16 vector add kernel on Modal H100")
    parser.add_argument("submission", help="Path to submission.py")
    parser.add_argument("-o", "--output", default="results.json")
    parser.add_argument(
        "--mode",
        choices=["test", "leaderboard"],
        default="leaderboard",
        help="'test' for correctness only, 'leaderboard' for correctness + benchmark",
    )
    args = parser.parse_args()

    try:
        with open(args.submission) as f:
            kernel_code = f.read()
    except FileNotFoundError:
        print(f"Error: {args.submission} not found")
        sys.exit(1)

    print(f"Submitting {args.submission} to Modal H100 ({args.mode} mode)...")

    evaluate_kernel = modal.Function.from_name("vectoradd-kernel-eval", "evaluate_kernel")

    MODAL_TIMEOUT = 360
    result_holder = [None]
    error_holder = [None]

    def _call():
        try:
            result_holder[0] = evaluate_kernel.remote(kernel_code, mode=args.mode)
        except Exception as e:
            error_holder[0] = e

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=MODAL_TIMEOUT)

    if t.is_alive():
        print(f"Error: Modal call timed out after {MODAL_TIMEOUT}s", file=sys.stderr)
        sys.exit(2)
    if error_holder[0] is not None:
        print(f"Error: Modal call failed: {error_holder[0]}", file=sys.stderr)
        sys.exit(1)

    raw = result_holder[0]
    res = json.loads(raw)
    md = format_results_markdown(res, mode=args.mode)

    with open(args.output, "w") as f:
        json.dump(md, f)

    print(md)
    sys.exit(0 if res["success"] else 1)


if __name__ == "__main__":
    main()
