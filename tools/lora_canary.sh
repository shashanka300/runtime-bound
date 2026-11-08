#!/usr/bin/env bash
# LoRA canary on the scorer engine (patched vLLM 0.19.1, wheelhouse of 2026-09-27): serve the
# starter main_lora plus a canary adapter (random lora_B), compare against the base model, then
# restore the standard server.
#   bash tools/lora_canary.sh [label]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
LABEL="${1:-v019_20260927}"
LOGDIR=data/logs
say() { echo "$(date -u +%FT%TZ) $*"; }
vllm_pids() { pgrep -f "bin/vllm serve google/gemma-4-31B" || true; }
stop_vllm() {
  local pids; pids=$(vllm_pids)
  if [[ -n "$pids" ]]; then
    kill $pids 2>/dev/null
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
restore_standard() {
  curl -sf localhost:8000/v1/models 2>/dev/null | grep -q main_lora && curl -sf localhost:8000/v1/models | grep -q '"gemma-4-31b-it"' \
    && ! curl -sf localhost:8000/v1/models | grep -q canary && { say "standard server already serving"; return; }
  say "restoring vLLM 0.30 + starter LoRA"; stop_vllm
  setsid -f bash infra/vllm/serve_submission.sh > "$LOGDIR/serve_restore.out" 2>&1 < /dev/null
  if wait_healthy 900; then say "standard server healthy"; else say "ERROR: standard server not healthy"; fi
}

infra/vllm019/.venv/bin/python tools/lora_canary.py make
trap restore_standard EXIT
stop_vllm
say "starting vLLM 0.19.1 with LoRA"
setsid -f bash infra/vllm019/serve.sh --enable-lora --max-loras 2 --max-lora-rank 8 \
  --lora-modules main_lora=data/competition/sample_submission/adapters/main_lora \
  canary=data/probes/lora_canary > "$LOGDIR/serve019_lora.out" 2>&1 < /dev/null
wait_healthy 900 || { say "ERROR: vLLM 0.19.1 with LoRA not healthy"; tail -30 "$LOGDIR/serve019_lora.out"; exit 1; }
curl -s localhost:8000/v1/models | python3 -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"]])'
infra/vllm019/.venv/bin/python tools/lora_canary.py compare "$LABEL"
say "canary done"
