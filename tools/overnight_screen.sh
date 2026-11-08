#!/usr/bin/env bash
# Overnight leaderboard screen (plan docs/archive/PLAN_NOV.md, B1/B3), Kaggle-matched: official harness
# (wheelhouse V31) + vLLM 0.19.1 with the scorer's V31 flags, one task at a time, per-task time
# cap scaled by 1/1.6 for the slower 4xL4 scorer (7 min -> 4.375 min). Variants run in order on
# the same task list (officially gradable public tasks minus the slow-grading rich_4006/rich_4076).
#   setsid -f bash tools/overnight_screen.sh > data/logs/overnight_screen.out 2>&1 < /dev/null
# Resumable: tools/official_screen.py skips tasks already in results.jsonl.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
LOGDIR=data/logs
OUT=${OUT:-data/screen_nov}
TASKS=${TASKS:-data/competition/tasks.jsonl}
VARIANTS=(${VARIANTS:-lb04_public_7min lb05a_think2048 lb05b_audit_rules})
SCALE=0.625
say() { echo "$(date -u +%FT%TZ) $*"; }
vllm_pids() { pgrep -f "bin/vllm serve google/gemma-4-3[1]B" || true; }

stop_vllm() {
  local pids
  pids=$(vllm_pids)
  if [[ -n "$pids" ]]; then
    say "stopping vLLM: $pids"
    kill $pids 2>/dev/null
    for _ in $(seq 60); do [[ -z "$(vllm_pids)" ]] && break; sleep 5; done
    pids=$(vllm_pids); [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null
  fi
  pgrep -f "VLLM::EngineCor[e]" | xargs -r kill -9 2>/dev/null
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
v019_up() { curl -sf localhost:8000/v1/models 2>/dev/null | grep -q gemma-4-31b-it-qat-w4a16-ct; }

start_v019() {
  v019_up && ! standard_up && { say "vLLM 0.19.1 already serving"; return 0; }
  stop_vllm
  say "starting vLLM 0.19.1 (V31 flags)"
  setsid -f bash infra/vllm019/serve.sh > "$LOGDIR/serve_v019.out" 2>&1 < /dev/null
  wait_healthy 1200 && v019_up
}

restore_standard() {
  standard_up && return
  say "restoring vLLM 0.30 + starter LoRA"
  stop_vllm
  setsid -f bash infra/vllm/serve_submission.sh > "$LOGDIR/serve_restore.out" 2>&1 < /dev/null
  if wait_healthy 900; then say "standard server healthy"; else say "ERROR: standard server not healthy"; fi
}

IDS=${IDS:-$(python3 - <<'EOF'
import csv
rows = csv.DictReader(open("leaderboard/official_gradability.csv"))
print(",".join(r["instance_id"] for r in rows
               if r["gradable"] == "true" and r["instance_id"] not in ("rich_4006", "rich_4076")))
EOF
)}
NTASK=$(echo "$IDS" | tr ',' '\n' | wc -l)
say "tasks: $NTASK; variants: ${VARIANTS[*]}"

start_v019 || { say "ERROR: vLLM 0.19.1 failed to start"; restore_standard; exit 1; }
[[ -n "${NO_RESTORE:-}" ]] || trap restore_standard EXIT

for v in "${VARIANTS[@]}"; do
  b=leaderboard/${BUNDLE:-$v}
  [[ -d "$b" ]] || { say "ERROR: missing bundle $b"; continue; }
  say "$v: start"
  infra/official/.venv/bin/python tools/official_screen.py "$b" "$OUT/$v" \
      --ids "$IDS" --time-scale "$SCALE" --tasks "$TASKS" >> "$LOGDIR/screen_$v.out" 2>&1 &
  pid=$!
  # watchdog: each task <= 4.4 min agent + grading; 12 min per task is a hard ceiling
  limit=$(( $(date +%s) + NTASK * 12 * 60 ))
  fails=0
  while kill -0 "$pid" 2>/dev/null; do
    if ! curl -sf localhost:8000/health >/dev/null; then
      fails=$((fails + 1)); say "WARNING: vLLM health failed during $v ($fails)"
      if (( fails >= 5 )); then
        say "restarting vLLM 0.19.1"; start_v019 || say "ERROR: restart failed"; fails=0
      fi
    else
      fails=0
    fi
    if (( $(date +%s) > limit )); then
      say "WATCHDOG: $v over time ceiling; stopping (rows kept)"
      kill -TERM "$pid" 2>/dev/null; sleep 20; kill -9 "$pid" 2>/dev/null
      break
    fi
    sleep 60
  done
  wait "$pid" 2>/dev/null
  n=$(wc -l < "$OUT/$v/results.jsonl" 2>/dev/null || echo 0)
  r=$(grep -c '"resolved": true' "$OUT/$v/results.jsonl" 2>/dev/null || echo 0)
  say "$v: done $n rows, $r resolved"
  docker ps -aq --filter "label=swegemma" | xargs -r docker rm -f >/dev/null 2>&1
done
say "screen finished"
