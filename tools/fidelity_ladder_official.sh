#!/usr/bin/env bash
# Transfer-gap ladder, revised 2026-09-27. vLLM 0.19.1 (the scorer's engine) rejects
# thinking_token_budget unless --reasoning-config is set, which the scorer does not do, so only the
# scorer's own thinking translation ("official") can run on it. 2x2 with arm F (0.30, "budget"):
#   step 2  arm F on vLLM 0.19.1, "official"  (the scorer's serving setup; server must be up)
#   step 3  arm F on vLLM 0.30,   "official"  (same thinking, standard engine: isolates the engine)
#   setsid -f bash tools/fidelity_ladder_official.sh
# The restore trap is armed from the start: this script is only launched while 0.19.1 is serving.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
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

wait_healthy() {
  local deadline=$(( $(date +%s) + $1 ))
  while (( $(date +%s) < deadline )); do
    curl -sf localhost:8000/health >/dev/null && return 0
    sleep 10
  done
  return 1
}

standard_up() {
  curl -sf localhost:8000/v1/models 2>/dev/null | grep -q main_lora
}

restore_standard() {
  standard_up && { say "standard server already serving"; return; }
  say "restoring vLLM 0.30 + starter LoRA"
  stop_vllm
  setsid -f bash infra/vllm/serve_submission.sh > "$LOGDIR/serve_restore.out" 2>&1 < /dev/null
  if wait_healthy 900; then say "standard server healthy"; else say "ERROR: standard server not healthy after 15 min"; fi
}
trap restore_standard EXIT

run_arm() {  # $1 = label, $2 = thinking translation
  local out pid limit fails=0
  out=$(bash tools/run_notebook.sh E1b_context_fixes "$1" "{\"params.arms\": [\"F\"], \"params.seeds\": [0], \"params.label\": \"$1\", \"params.thinking_translation\": \"$2\", \"params.reuse_arm_a\": \"<repo>/experiments/E1_starter_baseline/runs/20260925T115751Z-fa203996-b8e7/artifacts/swebox/t6min\", \"params.reuse\": {\"B\": \"<repo>/experiments/E1b_context_fixes/runs/20260925T210110Z-31559935-5542/artifacts/swebox/arm_B\"}, \"params.comparisons\": [[\"B\", \"F\"], [\"A\", \"F\"]]}")
  say "$1: $out"
  pid=$(echo "$out" | sed -E 's/pid=([0-9]+).*/\1/')
  setsid -f bash tools/autocommit_results.sh "$pid" 1800 > "$LOGDIR/autocommit_$1.log" 2>&1 < /dev/null
  limit=$(( $(date +%s) + 110 * 60 * 16 / 10 ))
  while kill -0 "$pid" 2>/dev/null; do
    if ! curl -sf localhost:8000/health >/dev/null; then
      fails=$((fails + 1)); say "WARNING: vLLM health check failed during $1 ($fails)"
    else
      fails=0
    fi
    if (( $(date +%s) > limit || fails >= 5 )); then
      say "WATCHDOG: stopping $1 (over 1.6x ETA or server down 5 min); results so far are kept"
      kill -TERM "$pid" 2>/dev/null; sleep 20; kill -9 "$pid" 2>/dev/null
      uv run python -c "from swebox import sandbox as sb; print('orphans removed:', sb.remove_orphans(sb.client()))"
      break
    fi
    sleep 60
  done
  sleep 90
  say "$1 finished"
}

wait_healthy 60 || { say "ERROR: no server up at start"; exit 1; }
curl -s localhost:8000/v1/models | grep -q main_lora && { say "ERROR: expected vLLM 0.19.1, found the standard server"; exit 1; }
run_arm armF_v019_official official
restore_standard
standard_up || { say "ERROR: standard server not up; skipping step 3"; exit 1; }
run_arm armF_v030_official official
say "ladder done"
