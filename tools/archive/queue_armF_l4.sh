#!/usr/bin/env bash
# Clean engine effect for a thinking-on agent (2026-09-27 correction): E1b arm F (no LoRA,
# thinking on, max_output 8192, scorer thinking translation) at L4 speed (3.1 min) and
# concurrency 1 (as the scorer; no KV stalls), first on vLLM 0.19.1, then on vLLM 0.30.
#   setsid -f bash tools/queue_armF_l4.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
LOGDIR=data/logs
say() { echo "$(date -u +%FT%TZ) $*"; }
vllm_pids() { pgrep -f "bin/vllm serve google/gemma-4-31B" || true; }
stop_vllm() {
  local pids; pids=$(vllm_pids)
  if [[ -n "$pids" ]]; then
    say "stopping vLLM: $pids"; kill $pids 2>/dev/null
    for _ in $(seq 60); do [[ -z "$(vllm_pids)" ]] && break; sleep 5; done
    pids=$(vllm_pids); [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null
  fi
  pgrep -f "VLLM::EngineCore" | xargs -r kill -9 2>/dev/null; sleep 10
}
wait_healthy() {
  local deadline=$(( $(date +%s) + $1 ))
  while (( $(date +%s) < deadline )); do curl -sf localhost:8000/health >/dev/null && return 0; sleep 10; done
  return 1
}
standard_up() { curl -sf localhost:8000/v1/models 2>/dev/null | grep -q main_lora; }
restore_standard() {
  standard_up && { say "standard server already serving"; return; }
  say "restoring vLLM 0.30 + starter LoRA"; stop_vllm
  setsid -f bash infra/vllm/serve_submission.sh > "$LOGDIR/serve_restore.out" 2>&1 < /dev/null
  if wait_healthy 900; then say "standard server healthy"; else say "ERROR: standard server not healthy"; fi
}
run_arm_f() {  # $1 = label
  local out pid limit fails=0
  out=$(bash tools/run_notebook.sh E1b_context_fixes "$1" "{\"params.arms\": [\"F\"], \"params.seeds\": [0], \"params.label\": \"$1\", \"params.thinking_translation\": \"official\", \"params.time_minutes\": 3.1, \"params.concurrency\": 1, \"params.reuse_arm_a\": \"<repo>/experiments/E1_starter_baseline/runs/20260925T115751Z-fa203996-b8e7/artifacts/swebox/t6min\", \"params.reuse\": {\"B\": \"<repo>/experiments/E1b_context_fixes/runs/20260925T210110Z-31559935-5542/artifacts/swebox/arm_B\", \"G\": \"<repo>/experiments/E1b_context_fixes/runs/20260927T074744Z-2f65a6f0-1e0d/artifacts/swebox/arm_G\"}, \"params.comparisons\": [[\"G\", \"F\"], [\"B\", \"F\"]]}")
  say "$1: $out"
  pid=$(echo "$out" | sed -E 's/pid=([0-9]+).*/\1/')
  setsid -f bash tools/autocommit_results.sh "$pid" 1800 > "$LOGDIR/autocommit_$1.log" 2>&1 < /dev/null
  limit=$(( $(date +%s) + 150 * 60 * 16 / 10 ))
  while kill -0 "$pid" 2>/dev/null; do
    if ! curl -sf localhost:8000/health >/dev/null; then fails=$((fails + 1)); say "WARNING: vLLM health check failed ($fails)"; else fails=0; fi
    if (( $(date +%s) > limit || fails >= 5 )); then
      say "WATCHDOG: stopping $1"; kill -TERM "$pid"; sleep 20; kill -9 "$pid" 2>/dev/null
      uv run python -c "from swebox import sandbox as sb; print('orphans removed:', sb.remove_orphans(sb.client()))"
      break
    fi
    sleep 60
  done
  sleep 90
  say "$1 finished"
}

trap restore_standard EXIT
stop_vllm
say "starting vLLM 0.19.1"
setsid -f bash infra/vllm019/serve.sh > "$LOGDIR/serve019.out" 2>&1 < /dev/null
wait_healthy 900 || { say "ERROR: vLLM 0.19.1 not healthy"; exit 1; }
run_arm_f armF_v019_l4
restore_standard
standard_up || { say "ERROR: standard server not up"; exit 1; }
run_arm_f armF_v030_l4
say "queue done"
