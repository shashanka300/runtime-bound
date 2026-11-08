#!/usr/bin/env bash
# LoRA canary before/after the scorer's vLLM patch: the same strong canary (lora_B ~ N(0,1) x 50)
# on the wheelhouse vLLM 0.19.1 of 2026-09-25 (adapters reported wiped) and of 2026-09-27
# (patched). Leaves infra/vllm019 on the 2026-09-27 wheel.
#   bash tools/lora_canary_before_after.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
export CANARY_SCALE=50
OLD=data/wheelhouse_20260925/vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl
NEW=data/wheelhouse/vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl
install() { uv pip install --python infra/vllm019/.venv/bin/python --no-deps --reinstall "$1" 2>&1 | tail -1; }
trap 'install "$NEW"' EXIT
echo "== old wheel (2026-09-25)"; install "$OLD"
bash tools/lora_canary.sh old_wheel_s50
echo "== new wheel (2026-09-27)"; install "$NEW"
bash tools/lora_canary.sh new_wheel_s50
echo "before/after done"
