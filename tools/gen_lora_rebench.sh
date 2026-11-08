#!/usr/bin/env bash
# LoRA data generation on the SWE-rebench tasks (tools/ingest_rebench.py): same passes as
# tools/gen_lora_data.sh (plain -> hint1 -> hint2, lb04 at the screen budgets), with the
# competition-shaped root data/external/rebench_root (COMP_DIR for the official harness,
# DATA_ROOT for swebox grading). Model traffic goes through tools/record_proxy.py.
#   setsid -f bash tools/gen_lora_rebench.sh > data/logs/gen_rebench.out 2>&1 < /dev/null
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
RROOT=$PWD/data/external/rebench_root
export COMP_DIR=$RROOT/competition
OUT=data/lora_rebench
BUNDLE=${BUNDLE:-lb04_public_7min}
say() { echo "$(TZ=Asia/Kolkata date '+%F %H:%M IST') $*"; }
mkdir -p "$OUT"
python3 tools/make_hinted_tasks.py --out "$OUT" --tasks "$COMP_DIR/tasks.jsonl" > /dev/null
python3 -c "import json;[print(json.loads(l)['instance_id']) for l in open('$COMP_DIR/tasks.jsonl')]" > "$OUT/pool_ids.txt"
say "pool: $(wc -l < "$OUT/pool_ids.txt") tasks"
# record every model request/response (exact prompts incl. compaction) for per-call SFT examples
uv run python tools/record_proxy.py --port 8001 --upstream http://127.0.0.1:8000   --out "$OUT/requests.jsonl" >> data/logs/record_proxy_rebench.out 2>&1 &
PROXY=$!
trap 'kill $PROXY 2>/dev/null' EXIT
sleep 5
export MODEL_PROXY_URL=http://127.0.0.1:8001/v1

unsolved() {
  python3 - "$1" "$OUT" <<'PY'
import json, sys, glob
solved = set()
for f in glob.glob(sys.argv[2] + "/gen_*/regrade.jsonl"):
    solved |= {json.loads(l)["instance_id"] for l in open(f) if json.loads(l)["resolved"]}
print("\n".join(i for i in open(sys.argv[1]).read().split() if i not in solved))
PY
}

run_pass() {
  local name=$1 tasks=$2 ids=$3 last=$4
  [[ -s "$ids" ]] || { say "$name: nothing to run"; return; }
  say "$name: $(wc -l < "$ids") tasks"
  OUT="$OUT" IDS="$(paste -sd, "$ids")" TASKS="$tasks" BUNDLE="$BUNDLE" VARIANTS="gen_$name" \
    NO_RESTORE=$([[ $last == 1 ]] && echo "" || echo 1) bash tools/overnight_screen.sh
  say "$name: regrading"
  DATA_ROOT=$RROOT uv run python tools/regrade_swebox.py "$OUT/gen_$name" --concurrency 3 > "data/logs/regrade_rb_$name.out" 2>&1
  say "$name: $(grep -c '"resolved": true' "$OUT/gen_$name/regrade.jsonl") solved (swebox)"
}

run_pass plain "$COMP_DIR/tasks.jsonl" "$OUT/pool_ids.txt" 0
unsolved "$OUT/pool_ids.txt" > "$OUT/ids_hint1.txt"
run_pass hint1 "$OUT/tasks_hint1.jsonl" "$OUT/ids_hint1.txt" 0
unsolved "$OUT/pool_ids.txt" > "$OUT/ids_hint2.txt"
run_pass hint2 "$OUT/tasks_hint2.jsonl" "$OUT/ids_hint2.txt" 1
say "rebench generation finished"
