#!/usr/bin/env bash
# User-level setup (no sudo). Idempotent; safe to re-run.
#   bash infra/bootstrap_user.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Research env (rbcore + notebooks + tracking UIs).
uv sync --all-packages
uv run python -m ipykernel install --user --name challeenge --display-name "challeenge (py3.12)"

# Serving env is isolated: vLLM pins its own torch/CUDA stack.
(cd infra/vllm && uv sync)

if [[ ! -d .git ]]; then
  git init -b main
fi
uv run pre-commit install
uv run nbstripout --install

echo "OK. Next: bash infra/data/download.sh   (needs Kaggle + HF credentials, see README)"
