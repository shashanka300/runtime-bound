# Ingest SWE-rebench tasks end to end (resumable); log: data/logs/ingest_rebench.out
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH=$HOME/.local/bin:$PATH
say() { echo "$(TZ=Asia/Kolkata date '+%F %H:%M IST') $*"; }
say "snapshots"; .venv/bin/python tools/ingest_rebench.py snapshots
[ -f data/external/rebench_root/competition/wheels_missing.txt ] || { say "wheels"; uv run --no-project --with pip python tools/ingest_rebench.py wheels; }
say "selfcheck gold"; uv run python tools/ingest_rebench.py selfcheck --which gold --concurrency 2
say "selfcheck empty"; uv run python tools/ingest_rebench.py selfcheck --which empty --concurrency 2
.venv/bin/python tools/ingest_rebench.py keep
du -sh data/external/rebench_root
say "ingest finished"
