#!/usr/bin/env bash
# Launch the EdgeGuard AI dashboard (run ./scripts/setup.sh first).
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -d .venv ]]; then
  echo "ERROR: .venv not found. Run ./scripts/setup.sh first." >&2; exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
exec streamlit run app.py
