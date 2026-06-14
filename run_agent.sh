#!/usr/bin/env bash
set -e
cd /workspace/mla_decode-advisor

# Source env for Modal/Anthropic credentials
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

SESSION="mla-agent"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
fi
tmux new-session -d -s "$SESSION" -c "/workspace/mla_decode-advisor" \
    "bash -c 'set -a && source /workspace/mla_decode-advisor/.env && set +a && uv run mla_decode/agent.py --baseline mla_decode/starting_point.py --iterations 25 2>&1 | tee /tmp/mla_agent_run.log; echo; echo \"--- agent finished, press any key to exit ---\"; read -n1'"
tmux attach-session -t "$SESSION"
