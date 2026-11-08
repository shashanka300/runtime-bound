#!/usr/bin/env bash
# V4 of the overnight screen: lb06_12b on vLLM 0.19.1 (V31 flags) serving the 12B model, same
# tasks and time scale as tools/overnight_screen.sh. Waits for that script to finish first.
#   setsid -f bash tools/screen_12b.sh > data/logs/screen_12b_chain.out 2>&1 < /dev/null
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"  # uv for infra/vllm/serve.sh (restore failed without it)
LOGDIR=data/logs
OUT=data/screen_nov
say() { echo "$(date -u +%FT%TZ) $*"; }
while pgrep -f "tools/overnight_scree[n].sh" >/dev/null; do sleep 60; done
say "overnight screen done; waiting for the 12B download"
while pgrep -f "hf downloa[d] google/gemma-4-12B" >/dev/null; do sleep 60; done
vllm_pids() { pgrep -f "bin/vllm serve google/gemma-4-" || true; }
stop_vllm() {
  local pids; pids=$(vllm_pids)
  [[ -n "$pids" ]] && { kill $pids 2>/dev/null; for _ in $(seq 60); do [[ -z "$(vllm_pids)" ]] && break; sleep 5; done; }
  pids=$(vllm_pids); [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null
  pgrep -f "VLLM::EngineCor[e]" | xargs -r kill -9 2>/dev/null
  sleep 10
}
wait_healthy() {
  local deadline=$(( $(date +%s) + $1 ))
  while (( $(date +%s) < deadline )); do curl -sf localhost:8000/health >/dev/null && return 0; sleep 10; done
  return 1
}
restore_standard() {
  curl -sf localhost:8000/v1/models 2>/dev/null | grep -q main_lora && return
  say "restoring vLLM 0.30 + starter LoRA"; stop_vllm
  setsid -f bash infra/vllm/serve_submission.sh > "$LOGDIR/serve_restore.out" 2>&1 < /dev/null
  wait_healthy 900 && say "standard server healthy" || say "ERROR: standard server not healthy"
}
trap 'kill $PROXY 2>/dev/null; restore_standard' EXIT
if pgrep -f "vllm serve google/gemma-4-12[B]" >/dev/null; then
  say "12B server already starting/serving"
else
  stop_vllm
  say "starting vLLM 0.19.1 with 12B"
  MODEL=google/gemma-4-12B-it-qat-w4a16-ct CONFIG=serve_12b.yaml HF_HUB_OFFLINE=1 \
    setsid -f bash infra/vllm019/serve.sh > "$LOGDIR/serve_v019_12b.out" 2>&1 < /dev/null
fi
wait_healthy 1200 || { say "ERROR: 12B server did not come up"; exit 1; }
# Track W: record every request/response so decision points can be replayed on edited weights
PATH="$HOME/.local/bin:$PATH" uv run python tools/record_proxy.py --port 8001 \
  --upstream http://127.0.0.1:8000 --out "$OUT/lb06_12b_requests.jsonl" \
  >> "$LOGDIR/record_proxy_12b.out" 2>&1 &
PROXY=$!
sleep 8
curl -sf localhost:8001/v1/models >/dev/null || { say "ERROR: record proxy not up"; exit 1; }
export MODEL_PROXY_URL=http://127.0.0.1:8001/v1
curl -sf localhost:8000/v1/models | grep -q gemma-4-12b-it-qat-w4a16-ct || { say "ERROR: wrong model served"; exit 1; }
IDS=$(python3 - <<'EOF'
import csv
rows = csv.DictReader(open("leaderboard/official_gradability.csv"))
print(",".join(r["instance_id"] for r in rows
               if r["gradable"] == "true" and r["instance_id"] not in ("rich_4006", "rich_4076")))
EOF
)
say "lb06_12b: start"
timeout $(( 39 * 12 * 60 )) infra/official/.venv/bin/python tools/official_screen.py leaderboard/lb06_12b \
  "$OUT/lb06_12b" --ids "$IDS" --time-scale 0.625 >> "$LOGDIR/screen_lb06_12b.out" 2>&1
r=$(grep -c '"resolved": true' "$OUT/lb06_12b/results.jsonl" 2>/dev/null || echo 0)
say "lb06_12b: done $(wc -l < "$OUT/lb06_12b/results.jsonl") rows, $r resolved"
