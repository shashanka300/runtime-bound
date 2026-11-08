#!/usr/bin/env bash
# Time-budget curve for the lb03 configuration (E1b arm G: no LoRA, thinking off, max_output
# 4096, scorer thinking translation) on the scorer engine (vLLM 0.19.1) at concurrency 1:
# 1.0 and 2.0 min per task (3.1 min is run 20260927T074744Z-2f65a6f0-1e0d). Used to read lb03's
# public score as an effective per-task budget on the 4xL4 box (e.g. prefill slower than modelled).
#   setsid -f bash tools/queue_g_budget.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
LOGDIR=data/logs
REUSE_G=$PWD/experiments/E1b_context_fixes/runs/20260927T074744Z-2f65a6f0-1e0d/artifacts/swebox/arm_G
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
run_g() {  # $1 = label, $2 = minutes, $3 = ETA minutes
  local out pid limit fails=0
  out=$(bash tools/run_notebook.sh E1b_context_fixes "$1" "{\"params.arms\": [\"G\"], \"params.seeds\": [0], \"params.label\": \"$1\", \"params.thinking_translation\": \"official\", \"params.time_minutes\": $2, \"params.concurrency\": 1, \"params.reuse_arm_a\": \"<repo>/experiments/E1_starter_baseline/runs/20260925T115751Z-fa203996-b8e7/artifacts/swebox/t6min\", \"params.reuse\": {\"B\": \"<repo>/experiments/E1b_context_fixes/runs/20260925T210110Z-31559935-5542/artifacts/swebox/arm_B\"}, \"params.comparisons\": [[\"B\", \"G\"]]}")
  say "$1: $out"
  pid=$(echo "$out" | sed -E 's/pid=([0-9]+).*/\1/')
  setsid -f bash tools/autocommit_results.sh "$pid" 1800 > "$LOGDIR/autocommit_$1.log" 2>&1 < /dev/null
  limit=$(( $(date +%s) + $3 * 60 * 16 / 10 ))
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
# G_BUDGETS: space-separated minutes to run (default: both); e.g. G_BUDGETS=2.0 after a restart.
for m in ${G_BUDGETS:-1.0 2.0}; do
  run_g "armG_v019_t${m%%.*}" "$m" $(awk "BEGIN{print int($m*30+30)}")
done
say "queue done"
