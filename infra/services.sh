#!/usr/bin/env bash
# Local observability services. Both are optional: runs always write their own
# files (JSONL/Parquet) and these UIs only index them.
#   bash infra/services.sh start|stop|status
#   MLflow  -> http://localhost:5000   (runs, params, metrics, artifacts)
#   Phoenix -> http://localhost:6006   (agent traces via OTLP)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
PIDS="$DATA_ROOT/services"
mkdir -p "$PIDS" "$DATA_ROOT/mlflow" "$DATA_ROOT/phoenix"
cd "$ROOT"

start() {
  if ! curl -sf localhost:5000/health >/dev/null; then
    nohup uv run mlflow server --host 127.0.0.1 --port 5000 \
      --backend-store-uri "sqlite:///$DATA_ROOT/mlflow/mlflow.db" \
      --artifacts-destination "$DATA_ROOT/mlflow/artifacts" \
      >"$PIDS/mlflow.log" 2>&1 & echo $! >"$PIDS/mlflow.pid"
  fi
  if ! curl -sf localhost:6006 >/dev/null; then
    PHOENIX_WORKING_DIR="$DATA_ROOT/phoenix" nohup uv run phoenix serve \
      >"$PIDS/phoenix.log" 2>&1 & echo $! >"$PIDS/phoenix.pid"
  fi
  status
}
stop() {
  for s in mlflow phoenix; do
    [[ -f "$PIDS/$s.pid" ]] && kill "$(cat "$PIDS/$s.pid")" 2>/dev/null || true
    rm -f "$PIDS/$s.pid"
  done
}
status() {
  sleep 2
  curl -sf localhost:5000/health >/dev/null && echo "mlflow  up  :5000" || echo "mlflow  starting/down (see $PIDS/mlflow.log)"
  curl -sf localhost:6006 >/dev/null && echo "phoenix up  :6006" || echo "phoenix starting/down (see $PIDS/phoenix.log)"
}
"${1:-status}"
