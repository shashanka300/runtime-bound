#!/usr/bin/env bash
# Commit and push experiment RESULTS (tables, figures) every INTERVAL seconds while a
# process is alive, then once more after it exits. Code changes are never committed here.
#   setsid -f bash tools/autocommit_results.sh <pid-to-follow> [interval_s]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
FOLLOW="$1"; INTERVAL="${2:-1800}"

commit_results() {
  bash tools/archive_notebooks.sh
  git add experiments/*/tables experiments/*/figures experiments/*/executed 2>/dev/null
  git diff --cached --quiet && return 0
  local changed
  changed=$(git diff --cached --name-only | sed -E 's#experiments/([^/]+)/.*#\1#' | sort -u | paste -sd, -)
  if git commit -q -m "results: update ${changed} (auto)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"; then
    git push -q && echo "$(date -u) pushed results: $changed"
  else
    git reset -q  # hooks rejected: leave the files for a manual review
    echo "$(date -u) commit rejected by hooks; left unstaged"
  fi
}

last=0
while kill -0 "$FOLLOW" 2>/dev/null; do
  if (( SECONDS - last >= INTERVAL )); then
    commit_results
    last=$SECONDS
  fi
  sleep 30  # notice the run's exit within 30 s, so the final commit is not delayed
done
commit_results
echo "$(date -u) followed process $FOLLOW exited; autocommit stopped"
