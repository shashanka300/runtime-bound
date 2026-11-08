#!/usr/bin/env bash
# Run an experiment notebook headless and fully detached (survives the launching shell).
#   bash tools/run_notebook.sh E1_starter_baseline my_label '{"params.seeds": [0, 1, 2]}'
# - syncs the jupytext pair first, so the .ipynb can never lag behind the edited .py
# - writes the executed notebook to data/queue (fast ext4; never the D: archive)
# - autosaves rarely; console logging stays at WARNING (full logs are in the run dir)
# Prints the papermill PID to watch with `kill -0 <pid>`.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="$1"; LABEL="$2"; OVERRIDES="${3:-{\}}"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
NB="experiments/$EXP/$EXP"
uv run jupytext --sync "$NB.py" >/dev/null
mkdir -p data/queue data/logs
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="data/queue/${EXP}_${LABEL}_${STAMP}.ipynb"
LOG="data/logs/${EXP}_${LABEL}_${STAMP}.log"
printf '%s' "$OVERRIDES" > "data/queue/.overrides_${STAMP}.json"
setsid -f bash -c "uv run papermill '$NB.ipynb' '$OUT' --cwd 'experiments/$EXP' \
  --autosave-cell-every 900 -p CONFIG_OVERRIDES \"\$(cat data/queue/.overrides_${STAMP}.json)\" \
  > '$LOG' 2>&1" < /dev/null > /dev/null 2>&1  # release the caller's pipe: $(...) returns now
for _ in $(seq 30); do
  PID=$(pgrep -f "^$ROOT/.venv/bin/python .*papermill .*${STAMP}" || true)
  [[ -n "$PID" ]] && break
  sleep 1
done
echo "pid=${PID:-?} log=$LOG out=$OUT"
