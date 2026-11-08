#!/usr/bin/env bash
# One-line progress snapshot of all background work, appended every INTERVAL seconds to
# data/logs/progress.log (default 600 s), so progress is tracked consistently even when nobody
# is watching. Leaderboard status is refreshed every 3rd snapshot.
#   setsid -f bash tools/progress_log.sh [interval_s]
# Stop: pkill -f 'progress_lo[g].sh'
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
INTERVAL="${1:-600}"
LOG=data/logs/progress.log
n=0
lb="lb:?"
while true; do
  ts=$(TZ=Asia/Kolkata date '+%Y-%m-%d %H:%M IST')
  if curl -sf -m 5 localhost:8000/v1/models >/dev/null; then
    if curl -sf -m 5 localhost:8000/v1/models | grep -q main_lora; then vllm="vllm0.30"; else vllm="vllm0.19.1"; fi
  else
    vllm="vllm:DOWN"
  fi
  jobs=""
  for nb in $(pgrep -af 'bin/[p]apermill' | grep -o 'data/queue/[^ ]*\.ipynb' | sort -u); do
    label=$(basename "$nb" .ipynb)
    exp=${label%%_20*}; exp=${exp%_*}
    jobs+=" [$label]"
  done
  runs=""
  for f in $(ls -dt experiments/*/runs/*/artifacts/swebox/*/task_results.jsonl 2>/dev/null | head -2); do
    age=$(( $(date +%s) - $(stat -c %Y "$f") ))
    if (( age < 1800 )); then
      d=$(dirname "$f"); rid=$(echo "$d" | cut -d/ -f4 | cut -c1-15)
      runs+=" $(basename "$d")@$rid:$(wc -l < "$f")rows/$(grep -c '"resolved": true' "$f")ok"
    fi
  done
  sweep=""
  pgrep -f 'official_gold_swee[p]' >/dev/null && sweep=" gold_sweep:$(wc -l < data/probes/official_gold.jsonl)/129"
  for f in $(ls -dt data/screen_nov/*/results.jsonl 2>/dev/null | head -1); do
    runs+=" screen:$(basename "$(dirname "$f")"):$(wc -l < "$f")/39rows/$(grep -c '"resolved": true' "$f")ok"
  done
  if (( n % 3 == 0 )); then
    lb=$(.venv/bin/kaggle competitions submissions -c gemma-4-developer-agent --csv 2>/dev/null \
      | python3 tools/lb_status.py 2>/dev/null || echo "lb:?")
  fi
  containers=$(docker ps -q 2>/dev/null | wc -l)
  echo "$ts | $vllm | containers:$containers | jobs:${jobs:- none}${runs}${sweep} | $lb" >> "$LOG"
  n=$((n + 1))
  sleep "$INTERVAL"
done
