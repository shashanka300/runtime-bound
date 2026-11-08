#!/usr/bin/env bash
# Multi-turn engine comparison (2026-09-27). First-turn probes are identical on vLLM 0.30 and
# 0.19.1, yet agent runs on 0.19.1 generate ~2x tokens per call; so compare later turns:
#   1. record 4 real agent sessions through tools/record_proxy.py on vLLM 0.30 (lb02 config:
#      no LoRA, thinking on, scorer thinking translation, 3.1 min, one at a time)
#   2. replay the recorded requests greedily on vLLM 0.30, then on vLLM 0.19.1
#   3. restore the standard server
#   setsid -f bash tools/queue_replay.sh   (waits for tools/queue_fidelity.sh to finish)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
LOGDIR=data/logs
P=data/probes
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

while pgrep -f "bash tools/queue_fidelity.sh" >/dev/null; do sleep 30; done
sleep 30
restore_standard
standard_up || { say "ERROR: need the standard server to record"; exit 1; }

rm -f "$P/recorded.jsonl"
uv run python tools/record_proxy.py --port 8001 --out "$P/recorded.jsonl" > "$LOGDIR/record_proxy.out" 2>&1 &
proxy=$!
sleep 5
say "recording 4 sessions through the proxy"
timeout 2400 uv run python -m swebox eval --subset fastapi_14356,fastapi_12942,rich_3938,requests_6589 \
  --submission leaderboard/lb02_no_lora_t60 --thinking-translation official \
  --api-base http://127.0.0.1:8001/v1 --time-minutes 3.1 --concurrency 1 \
  --out "$P/rec_eval" > "$LOGDIR/record_eval.out" 2>&1 || say "WARNING: recording eval exited non-zero"
kill "$proxy" 2>/dev/null
say "recorded $(wc -l < "$P/recorded.jsonl") requests"

say "replay on vLLM 0.30"
uv run python tools/replay_requests.py "$P/recorded.jsonl" v030 --greedy > "$LOGDIR/replay_v030.out" 2>&1 || say "WARNING: replay v030 failed"

trap restore_standard EXIT
stop_vllm
setsid -f bash infra/vllm019/serve.sh > "$LOGDIR/serve019.out" 2>&1 < /dev/null
wait_healthy 900 || { say "ERROR: vLLM 0.19.1 not healthy"; exit 1; }
say "replay on vLLM 0.19.1"
uv run python tools/replay_requests.py "$P/recorded.jsonl" v019 --greedy > "$LOGDIR/replay_v019.out" 2>&1 || say "WARNING: replay v019 failed"
say "replay queue done"
