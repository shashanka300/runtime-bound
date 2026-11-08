#!/usr/bin/env bash
# Quality gate: lint, format, types, tests. Run before every commit / overnight queue.
#   bash tools/check.sh [--fix]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"

if [[ "${1:-}" == "--fix" ]]; then
  uv run ruff check --fix .
  uv run ruff format .
else
  uv run ruff check .
  uv run ruff format --check .
fi
uv run mypy
uv run pytest -p no:warnings
# experiment-local tests: each experiment's `src` package is imported from its own folder
for t in experiments/*/tests; do
  (cd "$(dirname "$t")" && uv run pytest -q -p no:warnings -p no:cacheprovider tests)
done
