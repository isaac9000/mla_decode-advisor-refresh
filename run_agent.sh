#!/usr/bin/env bash
set -e
cd /workspace/mla_decode-advisor-refresh

# Source env for Modal/Anthropic credentials
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

echo "Checking GPU..."
OUTPUT=$(uv run python mla_decode/run_eval.py mla_decode/submission.py -o /tmp/gpu_check.json --mode test 2>&1)
echo "$OUTPUT"

GPU_LINE=$(echo "$OUTPUT" | grep "GPU:" || true)
echo ""
echo "Detected: $GPU_LINE"

if echo "$OUTPUT" | grep -q "NVIDIA H200"; then
    echo ""
    echo "--- GPU is H200 — launching agent in tmux ---"
    echo ""
    SESSION="mla-refresh-agent"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux kill-session -t "$SESSION"
    fi
    tmux new-session -d -s "$SESSION" -c "/workspace/mla_decode-advisor-refresh" \
        "bash -c 'set -a && source /workspace/mla_decode-advisor-refresh/.env && set +a && uv run mla_decode/agent.py --baseline mla_decode/starting_point.py --epoch-sizes 15 10 2>&1 | tee /tmp/mla_agent_refresh_run.log; echo; echo \"--- agent finished, press any key to exit ---\"; read -n1'"
    tmux attach-session -t "$SESSION"
else
    echo ""
    echo "--- GPU is NOT H200 — aborting ---"
    exit 1
fi
