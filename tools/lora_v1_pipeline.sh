#!/usr/bin/env bash
# LoRA v1, one training run (docs/design_changes.md 2026-11-08): after the rebench generation,
#  1. vLLM 0.19.1 canary: can it serve a rank-16 adapter on all Linears (MLP included) of the int4
#     checkpoint? (tools/lora_canary_all.py). If not, train attention-only.
#  2. build data/lora/sft_v2.jsonl from every dev and rebench pass;
#  3. memory smoke test at 8k (falls back to attention-only on failure);
#  4. train: lr 2e-4, 2 epochs, error-step masking, 5 runs held out, adapter saved per epoch.
#   setsid -f bash tools/lora_v1_pipeline.sh > data/logs/lora_v1.out 2>&1 < /dev/null
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
say() { echo "$(TZ=Asia/Kolkata date '+%F %H:%M IST') $*"; }
PY=infra/train/.venv/bin/python
ALL=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
ATTN=q_proj,k_proj,v_proj,o_proj
vllm_pids() { pgrep -f "bin/vllm serve google/gemma-4-3[1]B" || true; }
stop_vllm() {
  local pids; pids=$(vllm_pids)
  if [[ -n "$pids" ]]; then
    say "stopping vLLM: $pids"; kill $pids 2>/dev/null
    for _ in $(seq 60); do [[ -z "$(vllm_pids)" ]] && break; sleep 5; done
    pids=$(vllm_pids); [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null
  fi
  pgrep -f "VLLM::EngineCor[e]" | xargs -r kill -9 2>/dev/null; sleep 10
}
wait_healthy() {
  local deadline=$(( $(date +%s) + $1 ))
  while (( $(date +%s) < deadline )); do curl -sf localhost:8000/health >/dev/null && return 0; sleep 10; done
  return 1
}

say "waiting for the rebench generation to finish"
until grep -q "rebench generation finished" data/logs/gen_rebench.out; do sleep 60; done
while pgrep -f "[g]en_lora_rebench.sh" >/dev/null; do sleep 30; done
say "generation done"

# 1. serving canary
stop_vllm
infra/vllm019/.venv/bin/python tools/lora_canary_all.py make
setsid -f bash infra/vllm019/serve.sh --enable-lora --max-loras 1 --max-lora-rank 16 \
  --lora-modules canary_all=data/probes/lora_canary_all > data/logs/serve019_lora_all.out 2>&1 < /dev/null
TARGETS=$ATTN
if wait_healthy 900; then
  infra/vllm019/.venv/bin/python tools/lora_canary_all.py compare
  if python3 -c "import json,sys; sys.exit(0 if json.load(open('data/probes/lora_canary_all.json'))['mlp_lora_applied'] else 1)"; then
    TARGETS=$ALL; say "canary: vLLM applies the all-Linear adapter -> train on all Linears"
  else
    say "canary: MLP LoRA not applied by vLLM -> attention-only"
  fi
else
  say "canary: vLLM did not come up with the all-Linear adapter -> attention-only"
  tail -40 data/logs/serve019_lora_all.out
fi
stop_vllm

# 2. data
python3 tools/build_sft.py data/lora/gen_plain data/lora/gen_hint1 data/lora/gen_hint2 \
  data/lora_rebench/gen_plain data/lora_rebench/gen_hint1 data/lora_rebench/gen_hint2 --out data/lora/sft_v2.jsonl
say "sft_v2: $(wc -l < data/lora/sft_v2.jsonl) runs"

# 3. memory smoke (longest windows)
if ! $PY tools/train_lora.py data/lora/sft_v2.jsonl --max-len 8192 --cpu-offload --smoke --targets "$TARGETS"; then
  if [[ $TARGETS == "$ALL" ]]; then
    say "smoke failed with all Linears -> attention-only"; TARGETS=$ATTN
    $PY tools/train_lora.py data/lora/sft_v2.jsonl --max-len 8192 --cpu-offload --smoke --targets "$TARGETS" \
      || { say "ERROR: smoke failed with attention-only too"; exit 1; }
  else
    say "ERROR: smoke failed"; exit 1
  fi
fi

# 4. train
say "training: targets $TARGETS"
$PY tools/train_lora.py data/lora/sft_v2.jsonl --max-len 8192 --cpu-offload --targets "$TARGETS" \
  --out data/lora/adapter_v1 || { say "ERROR: training failed"; exit 1; }
say "lora v1 training finished"
