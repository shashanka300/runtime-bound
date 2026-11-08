#!/usr/bin/env bash
# Stop a papermill run once N tasks are finished unless it beats the baseline by --min-gain.
#   tools/early_stop.sh <papermill_pid> <results_dir> <baseline_dir> <n> [min_gain]
set -u
pid=$1 res=$2 base=$3 n=$4 gain=${5:-0.05}
export PATH=$HOME/.local/bin:$PATH
while kill -0 "$pid" 2>/dev/null; do
  done_n=$(wc -l < "$res/task_results.jsonl" 2>/dev/null || echo 0)
  if [ "$done_n" -ge "$n" ]; then
    if uv run python tools/early_stop.py "$res" "$base" --min-gain "$gain"; then
      echo "$(date -u +%FT%TZ) gain >= $gain at $done_n tasks: continuing"; exit 0
    fi
    echo "$(date -u +%FT%TZ) gain < $gain at $done_n tasks: stopping run"
    sid=$(ps -o sid= -p "$pid" | tr -d " ")
    pkill -TERM -s "$sid"; sleep 20; pkill -KILL -s "$sid" 2>/dev/null
    uv run python -c "from swebox import sandbox as sb; print(\"orphans removed:\", sb.remove_orphans(sb.client()))"
    exit 0
  fi
  sleep 60
done
echo "run ended before $n tasks"
