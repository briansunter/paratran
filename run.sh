#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run.sh [--model MODEL] [--model-dir DIR] [--host HOST] [--port PORT]
# Or via env vars: PARATRAN_MODEL, PARATRAN_MODEL_DIR

cd "$(dirname "$0")"

echo "Starting server..."
uv run paratran "$@"
