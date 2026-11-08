#!/usr/bin/env bash
# Build the scorer-engine environment: vLLM 0.19.1 and its serving dependencies at the exact
# versions of the official wheelhouse (Kaggle dataset metric/gemma-4-developer-agent-wheelhouse,
# downloaded to data/wheelhouse). torch and other packages absent from the wheelhouse come from
# PyPI (torch 2.10.0+cu128, which has sm_120 kernels for the RTX 5090).
#   bash infra/vllm019/setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="$HERE/../../data/wheelhouse"
export PATH="$HOME/.local/bin:$PATH"
cd "$HERE"
[[ -d .venv ]] || uv venv -q --python 3.12 .venv
# 1) resolve vLLM's full dependency tree (pulls torch etc.)
uv pip install --python .venv/bin/python --find-links "$W" "$W"/vllm-0.19.1-*.whl
# 2) pin everything the wheelhouse ships (the resolver may have picked newer versions, e.g.
#    transformers 5.17 breaks Gemma 4 config loading in vLLM 0.19.1); harness wheels are skipped
mapfile -t WHEELS < <(ls "$W"/*.whl | grep -vE '/(adk_|swegemma|google_adk|anthropic)')
uv pip install --python .venv/bin/python --no-deps --reinstall "${WHEELS[@]}"
.venv/bin/python -c 'import torch, vllm, transformers; print(torch.__version__, vllm.__version__, transformers.__version__)'
