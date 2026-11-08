#!/usr/bin/env bash
# Transfer-gap ladder on the scorer's engine (patched vLLM 0.19.1), after step 1 finishes:
#   step 2  E1b arm F on vLLM 0.19.1, thinking translation "budget" (our A1 mapping)
#   step 3  E1b arm F on vLLM 0.19.1, thinking translation "official" (the scorer's mapping:
#           no thinking budget reaches vLLM)
# then restores the standard vLLM 0.30 + starter-LoRA server.
#   setsid -f bash tools/fidelity_ladder.sh <pid-of-step-1-papermill>
# Safe to kill while waiting: the restore trap is armed only once this script has replaced the
# standard server. Watchdogs: server start 15 min; each run 1.6x its ~110 min ETA.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
WAIT_PID="${1:-}"
LOGDIR=data/logs
say() { echo "$(date -u +%FT%TZ) $*"; }

vllm_pids() { pgrep -f "bin/vllm serve google/gemma-4-31B" || true; }

stop_vllm() {
  local pids
  pids=$(vllm_pids)
  if [[ -n "$pids" ]]; then
    say "stopping vLLM: $pids"
    kill $pids 2>/dev/null
    for _ in $(seq 60); do [[ -z "$(vllm_pids)" ]] && break; sleep 5; done
    pids=$(vllm_pids); [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null
  fi
  pgrep -f "VLLM::EngineCore" | xargs -r kill -9 2>/dev/null
  sleep 10
}

wait_healthy() {  # $1 = seconds
  local deadline=$(( $(date +%s) + $1 ))
  while (( $(date +%s) < deadline )); do
    curl -sf localhost:8000/health >/dev/null && return 0
    sleep 10
  done
  return 1
}

restore_standard() {
  say "restoring vLLM 0.30 + starter LoRA"
  stop_vllm
  setsid -f bash infra/vllm/serve_submission.sh > "$LOGDIR/serve_restore.out" 2>&1 < /dev/null
  if wait_healthy 900; then say "standard server healthy"; else say "ERROR: standard server not healthy after 15 min"; fi
}

run_arm() {  # $1 = label, $2 = thinking translation
  local out pid limit
  out=$(bash tools/run_notebook.sh E1b_context_fixes "$1" "{\"params.arms\": [\"F\"], \"params.seeds\": [0], \"params.label\": \"$1\", \"params.thinking_translation\": \"$2\", \"params.reuse_arm_a\": \"<repo>/experiments/E1_starter_baseline/runs/20260925T115751Z-fa203996-b8e7/artifacts/swebox/t6min\", \"params.reuse\": {\"B\": \"<repo>/experiments/E1b_context_fixes/runs/20260925T210110Z-31559935-5542/artifacts/swebox/arm_B\"}, \"params.comparisons\": [[\"B\", \"F\"], [\"A\", \"F\"]]}")
  say "$1: $out"
  pid=$(echo "$out" | sed -E 's/pid=([0-9]+).*/\1/')
  setsid -f bash tools/autocommit_results.sh "$pid" 1800 > "$LOGDIR/autocommit_$1.log" 2>&1 < /dev/null
  limit=$(( $(date +%s) + 110 * 60 * 16 / 10 ))
  while kill -0 "$pid" 2>/dev/null; do
    if ! curl -sf localhost:8000/health >/dev/null; then say "WARNING: vLLM health check failed during $1"; fi
    if (( $(date +%s) > limit )); then
      say "WATCHDOG: $1 exceeded 1.6x ETA; stopping it (results so far are kept)"
      kill -TERM -- -"$(ps -o sid= -p "$pid" | tr -d ' ')" 2>/dev/null; sleep 20
      uv run python -c "from swebox import sandbox as sb; print('orphans removed:', sb.remove_orphans(sb.client()))"
      break
    fi
    sleep 60
  done
  sleep 90  # let the autocommit notice the exit and push
  say "$1 finished"
}

if [[ -n "$WAIT_PID" ]]; then
  say "waiting for step 1 (pid $WAIT_PID)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  sleep 90
fi

trap restore_standard EXIT  # from here on this script owns the server
stop_vllm
say "starting vLLM 0.19.1"
setsid -f bash infra/vllm019/serve.sh > "$LOGDIR/serve019.out" 2>&1 < /dev/null
if ! wait_healthy 900; then
  say "ERROR: vLLM 0.19.1 not healthy after 15 min; tail of log:"; tail -40 "$LOGDIR/serve019.out"
  exit 1
fi
say "vLLM 0.19.1 healthy; tool-call smoke test:"
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "gemma-4-31b-it", "max_tokens": 1024,
  "chat_template_kwargs": {"enable_thinking": true},
  "messages": [{"role": "user", "content": "List the files in /workspace."}],
  "tools": [{"type": "function", "function": {"name": "run_command",
    "description": "Run a shell command", "parameters": {"type": "object",
    "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
}' | python3 -c 'import json,sys; m=json.load(sys.stdin)["choices"][0]["message"]; print("tool_calls:", m.get("tool_calls")); print("content:", (m.get("content") or "")[:200])'

run_arm armF_v019 budget
run_arm armF_v019_official official
say "ladder steps 2-3 done"
