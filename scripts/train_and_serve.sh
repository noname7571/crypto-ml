#!/usr/bin/env bash
set -euo pipefail

# One-command local runner: train a model, then serve API with auto-loaded bundle.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-xgboost}"
DATA_PATH="${DATA_PATH:-data/raw/BTCUSDT_1h.parquet}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Virtual environment not found at .venv."
  echo "Create it with: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

source .venv/bin/activate

echo "[1/2] Training model (${MODEL})..."
"$PYTHON_BIN" -m src.training.train --model "$MODEL" --data-path "$DATA_PATH"

echo "[2/2] Starting API on ${HOST}:${PORT}..."
exec uvicorn src.api.app:app --host "$HOST" --port "$PORT"
