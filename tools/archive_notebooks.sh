#!/usr/bin/env bash
# Copy executed notebooks (outputs and graphs included) from data/queue into
# experiments/<exp>/executed/ so they are versioned. Never overwrites: names are unique per run
# (<exp>_<label>_<UTC stamp>.ipynb). Skips notebooks still being written (papermill running on
# them) and oversized ones from crashed runs (> MAX_KB, e.g. the OOM log-flood notebook).
#   bash tools/archive_notebooks.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MAX_KB="${MAX_KB:-2000}"
for nb in data/queue/*.ipynb; do
  [[ -e "$nb" ]] || continue
  name=$(basename "$nb")
  pgrep -f "papermill .*${name}" >/dev/null && continue
  (( $(du -k "$nb" | cut -f1) > MAX_KB )) && continue
  for exp_dir in experiments/*/; do
    exp=$(basename "$exp_dir")
    if [[ "$name" == "${exp}_"* ]]; then
      mkdir -p "${exp_dir}executed"
      [[ -e "${exp_dir}executed/$name" ]] || cp -p "$nb" "${exp_dir}executed/$name"
      break
    fi
  done
done
