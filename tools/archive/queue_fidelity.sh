#!/usr/bin/env bash
# Evening queue, 2026-09-27 (after the L4-speed G/H run):
#   1. engine probe on the standard server (vLLM 0.30)       data/probes/v030.jsonl
#   2. swap to the scorer engine (vLLM 0.19.1), engine probe   data/probes/v019.jsonl
#   3. E5 phase_v2_3 under Kaggle fidelity (no LoRA, official thinking, 3.1 min, one at a time)
#   4. E5 phase_v2_4 (thinking off in every phase), same setup
#   5. restore the standard server (EXIT trap, armed at the swap)
# Both E5 runs compare against E1b arm G at L4 speed (the setup of lb03).
#   setsid -f bash tools/queue_fidelity.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
LOGDIR=data/logs
BASELINE=$PWD/experiments/E1b_context_fixes/runs/20260927T074744Z-2f65a6f0-1e0d/artifacts/swebox/arm_G
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

run_e5() {  # $1 = variant, $2 = label
  local out pid limit fails=0
  out=$(bash tools/run_notebook.sh E5_context_architecture "$2" "{\"params.variant\": \"$1\", \"params.subset\": \"dev40\", \"params.seeds\": [0], \"params.label\": \"$2\", \"params.adapter\": null, \"params.thinking_translation\": \"official\", \"params.time_minutes\": 3.1, \"params.concurrency\": 1, \"params.baseline_dir\": \"$BASELINE\", \"params.baseline_label\": \"G (thinking off, L4 speed)\"}")
  say "$2: $out"
  pid=$(echo "$out" | sed -E 's/pid=([0-9]+).*/\1/')
  setsid -f bash tools/autocommit_results.sh "$pid" 1800 > "$LOGDIR/autocommit_$2.log" 2>&1 < /dev/null
  limit=$(( $(date +%s) + 150 * 60 * 16 / 10 ))
  while kill -0 "$pid" 2>/dev/null; do
    if ! curl -sf localhost:8000/health >/dev/null; then
      fails=$((fails + 1)); say "WARNING: vLLM health check failed during $2 ($fails)"
    else
      fails=0
    fi
    if (( $(date +%s) > limit || fails >= 5 )); then
      say "WATCHDOG: stopping $2 (over 1.6x ETA or server down 5 min)"
      kill -TERM "$pid"; sleep 20; kill -9 "$pid" 2>/dev/null
      uv run python -c "from swebox import sandbox as sb; print('orphans removed:', sb.remove_orphans(sb.client()))"
      break
    fi
    sleep 60
  done
  sleep 90
  say "$2 finished"
}

while pgrep -f "bash tools/chain_arms_gh.sh" >/dev/null; do sleep 30; done
sleep 30
wait_healthy 900 || { say "ERROR: no server after the L4 run"; exit 1; }
if standard_up; then
  say "probe on vLLM 0.30"
  timeout 3600 uv run python tools/engine_probe.py v030 > "$LOGDIR/probe_v030.out" 2>&1 || say "WARNING: probe v030 failed"
  tail -1 "$LOGDIR/probe_v030.out"
fi

trap restore_standard EXIT
stop_vllm
say "starting vLLM 0.19.1"
setsid -f bash infra/vllm019/serve.sh > "$LOGDIR/serve019.out" 2>&1 < /dev/null
wait_healthy 900 || { say "ERROR: vLLM 0.19.1 not healthy after 15 min"; exit 1; }
say "vLLM 0.19.1 healthy; probe"
timeout 3600 uv run python tools/engine_probe.py v019 > "$LOGDIR/probe_v019.out" 2>&1 || say "WARNING: probe v019 failed"
tail -1 "$LOGDIR/probe_v019.out"

run_e5 phase_v2_3 fid_l4_v2_3
run_e5 phase_v2_4 fid_l4_v2_4
say "queue done"
