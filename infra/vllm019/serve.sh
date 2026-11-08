#!/usr/bin/env bash
# Serve the 31B model with the scorer's own engine: the patched vLLM 0.19.1 from the official
# wheelhouse (Kaggle dataset metric/gemma-4-developer-agent-wheelhouse, in data/wheelhouse).
# Transfer-gap ladder step 2. Same served name and port as infra/vllm/serve.sh, so the harness
# needs no change; flags in serve.yaml mirror the scorer where one 32 GB GPU allows.
#   bash infra/vllm019/serve.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
MODEL="${MODEL:-google/gemma-4-31B-it-qat-w4a16-ct}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/hf}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# Note (2026-09-27): on one 32 GB card this engine reports 10,496 KV tokens (vs 45,506 for vLLM
# 0.30 on less KV memory; accounting differs), and at concurrency 2 long agent requests waited in
# the queue for minutes. Evaluate on it at concurrency 1 only, as the scorer does.
mkdir -p "$DATA_ROOT/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$DATA_ROOT/logs/vllm019_${STAMP}.log"
ln -sfn "$LOG" "$DATA_ROOT/logs/vllm_latest.log"
{ echo "MODEL=$MODEL"; echo "ENGINE=vllm-0.19.1 (official wheelhouse)"; echo "ARGS=$*";
  env | grep -E '^(VLLM_|CUDA_|HF_HUB_)' | sort; } >"$DATA_ROOT/logs/vllm019_${STAMP}.env"
ln -sfn "$DATA_ROOT/logs/vllm019_${STAMP}.env" "$DATA_ROOT/logs/vllm_latest.env"
echo "model=$MODEL  log=$LOG"
exec "$HERE/.venv/bin/vllm" serve "$MODEL" --config "$HERE/${CONFIG:-serve.yaml}" --no-scheduler-reserve-full-isl "$@" 2>&1 | tee "$LOG"
