#!/usr/bin/env bash
# LoRA data synthesis, generation passes (docs/design_changes.md 2026-11-06). Official harness +
# vLLM 0.19.1 with the lb04 bundle at the screen's budgets (28 calls, 7 min x 0.625), on the 90
# public tasks outside the 39-task screen (kept held out). Each pass is graded with swebox
# (tools/regrade_swebox.py); the next pass reruns only the tasks still unsolved:
#   plain  -> no hint
#   hint1  -> data/lora/tasks_hint1.jsonl (files of the reference fix)
#   hint2  -> data/lora/tasks_hint2.jsonl (files + symbols + size)
#   setsid -f bash tools/gen_lora_data.sh > data/logs/gen_lora.out 2>&1 < /dev/null
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
OUT=data/lora
BUNDLE=${BUNDLE:-lb04_public_7min}
say() { echo "$(TZ=Asia/Kolkata date '+%F %H:%M IST') $*"; }
mkdir -p "$OUT"
python3 tools/make_hinted_tasks.py --out "$OUT" > /dev/null

python3 - > "$OUT/pool_ids.txt" <<'PY'
import json
screen = {json.loads(l)["instance_id"] for l in open("data/screen_nov/lb04_public_7min/results.jsonl")}
for l in open("data/competition/tasks.jsonl"):
    i = json.loads(l)["instance_id"]
    if i not in screen:
        print(i)
PY
say "pool: $(wc -l < "$OUT/pool_ids.txt") tasks"

unsolved() {  # ids from $1 not resolved in any regrade so far
  python3 - "$1" <<'PY'
import json, sys, glob
solved = set()
for f in glob.glob("data/lora/gen_*/regrade.jsonl"):
    solved |= {json.loads(l)["instance_id"] for l in open(f) if json.loads(l)["resolved"]}
print("\n".join(i for i in open(sys.argv[1]).read().split() if i not in solved))
PY
}

run_pass() {  # name tasks_file ids_file last
  local name=$1 tasks=$2 ids=$3 last=$4
  [[ -s "$ids" ]] || { say "$name: nothing to run"; return; }
  say "$name: $(wc -l < "$ids") tasks"
  OUT="$OUT" IDS="$(paste -sd, "$ids")" TASKS="$tasks" BUNDLE="$BUNDLE" VARIANTS="gen_$name" \
    NO_RESTORE=$([[ $last == 1 ]] && echo "" || echo 1) bash tools/overnight_screen.sh
  say "$name: regrading"
  uv run python tools/regrade_swebox.py "$OUT/gen_$name" --concurrency 3 > "data/logs/regrade_$name.out" 2>&1
  say "$name: $(grep -c '"resolved": true' "$OUT/gen_$name/regrade.jsonl") solved (swebox)"
}

run_pass plain data/competition/tasks.jsonl "$OUT/pool_ids.txt" 0
unsolved "$OUT/pool_ids.txt" > "$OUT/ids_hint1.txt"
run_pass hint1 "$OUT/tasks_hint1.jsonl" "$OUT/ids_hint1.txt" 0
unsolved "$OUT/pool_ids.txt" > "$OUT/ids_hint2.txt"
run_pass hint2 "$OUT/tasks_hint2.jsonl" "$OUT/ids_hint2.txt" 1
say "generation finished; solved tasks: $(( $(wc -l < "$OUT/pool_ids.txt") - $(unsolved "$OUT/pool_ids.txt" | grep -c .) ))"
