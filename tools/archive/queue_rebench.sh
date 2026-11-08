#!/usr/bin/env bash
# After the dev-task generation and the SWE-rebench ingest: build SFT v1 from the dev-task passes,
# run a 2-step training smoke test on the GPU (memory/speed of LoRA on the QAT 31B), then start
# the rebench generation (tools/gen_lora_rebench.sh).
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH=$HOME/.local/bin:$PATH
say() { echo "$(TZ=Asia/Kolkata date '+%F %H:%M IST') $*"; }
while pgrep -f '[g]en_lora_data.sh' >/dev/null || pgrep -f '[i]ngest_rebench.sh' >/dev/null; do sleep 120; done
say "queue: dev generation and ingest done"
python3 tools/build_sft.py $(ls -d data/lora/gen_*) --out data/lora/sft_v1.jsonl
say "queue: training smoke test (stopping vLLM)"
pkill -f "bin/vllm serv[e]" ; sleep 20; pgrep -f "VLLM::EngineCor[e]" | xargs -r kill -9; sleep 10
HF_HUB_OFFLINE=1 timeout 3600 infra/train/.venv/bin/python tools/train_lora.py data/lora/sft_v1.jsonl --smoke \
  > data/logs/train_smoke.out 2>&1; say "smoke exit $?"; tail -5 data/logs/train_smoke.out
n=$(grep -c . data/external/rebench_root/competition/tasks.jsonl)
[[ -f data/external/rebench_root/competition/tasks_all.jsonl && $n -gt 0 ]] || { say "queue: ingest incomplete; restoring server"; setsid -f bash infra/vllm/serve_submission.sh > data/logs/serve_restore.out 2>&1 < /dev/null; exit 1; }
say "queue: starting rebench generation on $n tasks"
bash tools/gen_lora_rebench.sh
