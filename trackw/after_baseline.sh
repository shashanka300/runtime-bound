#!/usr/bin/env bash
# Track W: when the recorded 12B baseline (tools/screen_12b.sh) has finished, take the GPU for
#   1. the activation pass (trackw/acts.py, transformers 12B bf16);
#   2. the LoRA canary check on vLLM 0.19.1 + 12B (trackw/lora_verify.py);
# then give the GPU back to the standard server.
#   setsid -f bash trackw/after_baseline.sh > data/logs/trackw_after_baseline.out 2>&1 < /dev/null
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
R=data/screen_nov/lb06_12b_requests.jsonl
say() { echo "$(date -u +%FT%TZ) $*"; }
stop_vllm() {
  pkill -f "bin/vllm serve google/gemma-4-"
  sleep 30
  pgrep -f "VLLM::EngineCor[e]" | xargs -r kill -9
  sleep 5
}
wait_healthy() {
  local deadline=$(( $(date +%s) + $1 ))
  while (( $(date +%s) < deadline )); do curl -sf localhost:8000/health >/dev/null && return 0; sleep 10; done
  return 1
}
while pgrep -f "tools/screen_12[b].sh" >/dev/null; do sleep 60; done
say "baseline finished"
pkill -f "trackw/dump_token[s].py"
python3 trackw/decision_points.py "$R"
stop_vllm
say "activation pass"
HF_HUB_OFFLINE=1 infra/vllm019/.venv/bin/python trackw/acts.py "$R" --device cuda
say "activation pass exit $?"
say "LoRA canary on vLLM 0.19.1 + 12B"
MODEL=google/gemma-4-12B-it-qat-w4a16-ct CONFIG=serve_12b.yaml HF_HUB_OFFLINE=1 \
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  setsid -f bash infra/vllm019/serve.sh --enable-lora --max-loras 1 --max-lora-rank 8 \
  --lora-modules canary=data/trackw/canary12b > data/logs/serve_v019_12b_lora.out 2>&1 < /dev/null
if wait_healthy 1200; then
  HF_HUB_OFFLINE=1 infra/vllm019/.venv/bin/python trackw/lora_verify.py data/trackw/canary12b
  say "lora verify exit $?"
  say "loop probe (model vs thinking-budget enforcement)"
  python3 trackw/loop_probe.py "$R"
  say "loop probe exit $?"
else
  say "ERROR: 12B+LoRA server did not come up"; tail -5 data/logs/serve_v019_12b_lora.out
fi
# the 12B+LoRA server stays up for the base-rate evaluations and the editor session
say "queue done; 12B+LoRA server left running"
