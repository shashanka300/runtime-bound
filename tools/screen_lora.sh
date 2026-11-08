#!/usr/bin/env bash
# Screen the LoRA v1 adapters as the scorer would run them: for each epoch adapter (trained on Modal,
# tools/modal_train.py), fetch it from the Modal volume, build a submission bundle = lb04_public_7min +
# `adapter: main_lora` + adapters/main_lora, serve vLLM 0.19.1 (V31 flags) with the adapter, and run the
# official-harness screen on the 39 public gradable tasks (same ids, time scale and grading as the lb04
# screen; none of them is in the LoRA training data).
#   setsid -f bash tools/screen_lora.sh > data/logs/screen_lora.out 2>&1 < /dev/null
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$PATH"
LOGDIR=data/logs
OUT=data/screen_lora
SCALE=0.625
MODAL=~/modal-venv/bin/modal
say() { echo "$(TZ=Asia/Kolkata date '+%F %H:%M IST') $*"; }
vllm_pids() { pgrep -f "bin/vllm serve google/gemma-4-3[1]B" || true; }
stop_vllm() {
  local pids; pids=$(vllm_pids)
  if [[ -n "$pids" ]]; then
    kill $pids 2>/dev/null
    for _ in $(seq 60); do [[ -z "$(vllm_pids)" ]] && break; sleep 5; done
    pids=$(vllm_pids); [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null
  fi
  pgrep -f "VLLM::EngineCor[e]" | xargs -r kill -9 2>/dev/null; sleep 10
}
wait_healthy() {
  local deadline=$(( $(date +%s) + $1 ))
  while (( $(date +%s) < deadline )); do curl -sf localhost:8000/health >/dev/null && return 0; sleep 10; done
  return 1
}
IDS=$(python3 - <<'EOF'
import csv
rows = csv.DictReader(open("leaderboard/official_gradability.csv"))
print(",".join(r["instance_id"] for r in rows
               if r["gradable"] == "true" and r["instance_id"] not in ("rich_4006", "rich_4076")))
EOF
)
say "tasks: $(echo "$IDS" | tr ',' '\n' | wc -l)"

# user (11-08): wait for both epochs, screen only the one with the lower held-out validation loss
say "waiting for the epoch-2 adapter on Modal"
until $MODAL volume ls gemma4-lora adapter_v1/epoch2 2>/dev/null | grep -q adapter_config.json; do sleep 300; done
sleep 120  # the val loss line is printed after the save
VALS=$(sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g' $LOGDIR/modal_train.out | grep -o 'saved /vol/adapter_v1/epoch[0-9]*; val loss [0-9.]*')
if [[ $(echo "$VALS" | grep -c epoch) -lt 2 ]]; then
  APP=$(sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g' $LOGDIR/modal_train.out | grep -o 'ap-[A-Za-z0-9]*' | head -1)
  VALS=$(timeout 120 $MODAL app logs "$APP" 2>/dev/null | grep -o 'saved /vol/adapter_v1/epoch[0-9]*; val loss [0-9.]*')
fi
say "validation: $(echo "$VALS" | tr '\n' ' ')"
BEST=$(echo "$VALS" | sort -t' ' -k5 -g | head -1 | grep -o 'epoch[0-9]*' | tr -d 'epoch')
BEST=${BEST:-2}
say "best epoch by validation loss: $BEST"

for e in $BEST; do
  name=lb04_lora_v1_e$e
  dest=data/lora/adapter_v1_modal/epoch$e
  rm -rf "$dest"; mkdir -p "$(dirname "$dest")"
  $MODAL volume get gemma4-lora adapter_v1/epoch$e "$(dirname "$dest")/" --force > /dev/null
  [[ -f $dest/adapter_config.json && -f $dest/adapter_model.safetensors ]] || { say "ERROR: fetch of epoch $e failed"; ls -la "$dest"; continue; }
  say "$name: fetched $(du -sh "$dest" | cut -f1)"

  b=leaderboard/$name
  rm -rf "$b"; cp -r leaderboard/lb04_public_7min "$b"
  sed -i 's/^model: \(.*\)$/model: \1\nadapter: main_lora/' "$b/agent.yaml"
  mkdir -p "$b/adapters"; cp -r "$dest" "$b/adapters/main_lora"
  grep -q '^adapter: main_lora' "$b/agent.yaml" || { say "ERROR: bundle $b has no adapter line"; continue; }

  stop_vllm
  say "$name: starting vLLM 0.19.1 with the adapter"
  setsid -f bash infra/vllm019/serve.sh --enable-lora --max-loras 1 --max-lora-rank 16 \
    --lora-modules main_lora="$PWD/$b/adapters/main_lora" > "$LOGDIR/serve019_$name.out" 2>&1 < /dev/null
  if ! wait_healthy 1200 || ! curl -sf localhost:8000/v1/models | grep -q main_lora; then
    say "ERROR: vLLM with the adapter did not come up"; tail -30 "$LOGDIR/serve019_$name.out"; continue
  fi
  say "$name: screen start"
  infra/official/.venv/bin/python tools/official_screen.py "$b" "$OUT/$name" \
      --ids "$IDS" --time-scale "$SCALE" --tasks data/competition/tasks.jsonl >> "$LOGDIR/screen_$name.out" 2>&1
  n=$(wc -l < "$OUT/$name/results.jsonl" 2>/dev/null || echo 0)
  r=$(grep -c '"resolved": true' "$OUT/$name/results.jsonl" 2>/dev/null || echo 0)
  say "$name: done $n rows, $r resolved (lb04 base: 16/39)"
  docker ps -aq --filter "label=swegemma" | xargs -r docker rm -f >/dev/null 2>&1
done
stop_vllm
say "lora screen finished"
