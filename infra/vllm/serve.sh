#!/usr/bin/env bash
# Start the long-lived vLLM server that every notebook talks to over HTTP.
#   bash infra/vllm/serve.sh                      # 31B int4 (default)
#   MODEL=google/gemma-4-12B-it-qat-w4a16-ct bash infra/vllm/serve.sh --max-num-seqs 4
# Extra args are forwarded to `vllm serve` and override serve.yaml.
# Logs go to $DATA_ROOT/logs/vllm_<timestamp>.log (parsed by E0 for KV block counts).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
MODEL="${MODEL:-google/gemma-4-31B-it-qat-w4a16-ct}"

# Triton JIT-compiles its kernel launchers with the system C compiler (even in eager mode).
command -v gcc >/dev/null || { echo "gcc missing: run 'bash infra/setup_wsl.sh' (installs build-essential)"; exit 1; }

export HF_HOME="${HF_HOME:-$DATA_ROOT/hf}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
# FlashInfer's top-k/top-p sampler JIT-compiles with nvcc (full CUDA toolkit, not installed
# in WSL). vLLM's PyTorch sampler is used instead; sampling is <1% of step time.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
mkdir -p "$DATA_ROOT/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$DATA_ROOT/logs/vllm_${STAMP}.log"
ln -sfn "$LOG" "$DATA_ROOT/logs/vllm_latest.log"
# Server-side settings for run manifests (rbcore.manifest reads vllm_latest.env).
{ echo "MODEL=$MODEL"; echo "ARGS=$*"; env | grep -E '^(VLLM_|CUDA_|HF_HUB_)' | sort; } \
  >"$DATA_ROOT/logs/vllm_${STAMP}.env"
ln -sfn "$DATA_ROOT/logs/vllm_${STAMP}.env" "$DATA_ROOT/logs/vllm_latest.env"

echo "model=$MODEL  log=$LOG"
cd "$HERE"
exec uv run --frozen vllm serve "$MODEL" --config "$HERE/serve.yaml" "$@" 2>&1 | tee "$LOG"
