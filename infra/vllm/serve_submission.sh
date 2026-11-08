#!/usr/bin/env bash
# Serve the base model plus a submission's LoRA adapters, named as the submission expects
# (agent.yaml `adapter: main_lora` -> served model "main_lora").
#   bash infra/vllm/serve_submission.sh [submission_dir] [extra vllm args...]
# LoRA slots/rank are sized to the adapters actually present (the eval box preallocates
# 8 x rank 128, which does not fit next to the 31B model on one 32 GB card).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUB="$(cd "${1:-$ROOT/data/competition/sample_submission}" && pwd)"
shift || true

mapfile -t ARGS < <(python3 - "$SUB" <<'EOF'
import json, sys
from pathlib import Path
sub = Path(sys.argv[1])
adapters = sorted(p for p in (sub / "adapters").glob("*") if (p / "adapter_config.json").exists())
if not adapters:
    sys.exit(0)
rank = max(json.loads((p / "adapter_config.json").read_text())["r"] for p in adapters)
rank = next(r for r in (8, 16, 32, 64, 128, 256, 320, 512) if r >= rank)  # vLLM-supported
print("--enable-lora")
print("--max-loras"); print(len(adapters))
print("--max-lora-rank"); print(rank)
print("--lora-modules")
for p in adapters:
    print(f"{p.name}={p}")
EOF
)
echo "submission=$SUB lora_args=${ARGS[*]:-none}"
exec bash "$HERE/serve.sh" "${ARGS[@]}" "$@"
