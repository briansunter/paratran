#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run.sh [--model MODEL] [--model-dir DIR] [--host HOST] [--port PORT]
# Or via env vars: PARATRAN_MODEL, PARATRAN_MODEL_DIR

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Installing dependencies..."
.venv/bin/pip install -e . --quiet

echo "Starting server..."
.venv/bin/python server.py "$@"
