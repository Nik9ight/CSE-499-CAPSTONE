#!/usr/bin/env bash
# Defaults: I0–I11 × qata/MosMed × seeds 42/1337/2026, batch 32.
# PYTHON selects your existing CUDA-enabled Python environment.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/run_cumulative.py" "$@"
