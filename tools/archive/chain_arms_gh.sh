#!/usr/bin/env bash
# After the transfer-gap ladder: E1b arms G (thinking off, max_output 4096) and H (thinking on,
# max_output 4096), no LoRA, scorer thinking translation, dev40 seed 0, on the scorer's engine
# (vLLM 0.19.1): the ladder showed the engine, not the translation, drives the gap
# (0.30/official ~ 0.30/budget per call; 0.19.1 generates ~2x tokens per call).
#   setsid -f bash tools/chain_arms_gh.sh
# Env overrides (defaults = the 2026-09-27 run): LABEL, TIME_MINUTES, CONCURRENCY, ETA_MIN.
# L4-speed run: LABEL=armsGH_v019_l4 TIME_MINUTES=3.1 CONCURRENCY=1 ETA_MIN=280
#   (the scorer's 5 min on 4xL4 decoding ~1.6x slower than one RTX 5090, tasks run one at a time)
# Safe to kill while waiting (the restore trap is armed only after the server swap).
# Watchdogs: server start 15 min; run 1.6x its ETA (results so far are kept).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
LOGDIR=data/logs
LABEL="${LABEL:-armsGH_v019}"
TIME_MINUTES="${TIME_MINUTES:-6.0}"
CONCURRENCY="${CONCURRENCY:-2}"
ETA_MIN="${ETA_MIN:-200}"
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

wait_healthy() {
  local deadline=$(( $(date +%s) + $1 ))
  while (( $(date +%s) < deadline )); do
    curl -sf localhost:8000/health >/dev/null && return 0
    sleep 10
  done
  return 1
}

standard_up() { curl -sf localhost:8000/v1/models 2>/dev/null | grep -q main_lora; }

restore_standard() {
  standard_up && { say "standard server already serving"; return; }
  say "restoring vLLM 0.30 + starter LoRA"
  stop_vllm
  setsid -f bash infra/vllm/serve_submission.sh > "$LOGDIR/serve_restore.out" 2>&1 < /dev/null
  if wait_healthy 900; then say "standard server healthy"; else say "ERROR: standard server not healthy after 15 min"; fi
}

while pgrep -f "bash tools/fidelity_ladder_official.sh" >/dev/null; do sleep 30; done
sleep 60
trap restore_standard EXIT
stop_vllm
say "starting vLLM 0.19.1"
setsid -f bash infra/vllm019/serve.sh > "$LOGDIR/serve019.out" 2>&1 < /dev/null
wait_healthy 900 || { say "ERROR: vLLM 0.19.1 not healthy after 15 min"; tail -30 "$LOGDIR/serve019.out"; exit 1; }
say "vLLM 0.19.1 healthy"

overrides="{\"params.arms\": [\"G\", \"H\"], \"params.seeds\": [0], \"params.label\": \"$LABEL\", \"params.thinking_translation\": \"official\", \"params.time_minutes\": $TIME_MINUTES, \"params.concurrency\": $CONCURRENCY, \"params.reuse_arm_a\": \"<repo>/experiments/E1_starter_baseline/runs/20260925T115751Z-fa203996-b8e7/artifacts/swebox/t6min\", \"params.reuse\": {\"B\": \"<repo>/experiments/E1b_context_fixes/runs/20260925T210110Z-31559935-5542/artifacts/swebox/arm_B\"}, \"params.comparisons\": [[\"B\", \"G\"], [\"B\", \"H\"], [\"G\", \"H\"]]}"
out=$(bash tools/run_notebook.sh E1b_context_fixes "$LABEL" "$overrides")
say "$LABEL: $out"
pid=$(echo "$out" | sed -E 's/pid=([0-9]+).*/\1/')
setsid -f bash tools/autocommit_results.sh "$pid" 1800 > "$LOGDIR/autocommit_$LABEL.log" 2>&1 < /dev/null
limit=$(( $(date +%s) + ETA_MIN * 60 * 16 / 10 ))
fails=0
while kill -0 "$pid" 2>/dev/null; do
  if ! curl -sf localhost:8000/health >/dev/null; then
    fails=$((fails + 1)); say "WARNING: vLLM health check failed ($fails)"
  else
    fails=0
  fi
  if (( $(date +%s) > limit || fails >= 5 )); then
    say "WATCHDOG: stopping $LABEL (over 1.6x ETA or server down 5 min)"
    kill -TERM "$pid"; sleep 20; kill -9 "$pid" 2>/dev/null
    uv run python -c "from swebox import sandbox as sb; print('orphans removed:', sb.remove_orphans(sb.client()))"
    break
  fi
  sleep 60
done
sleep 90
say "$LABEL finished"
