#!/usr/bin/env bash
# One-time setup for macOS: project-local virtualenv, dependencies, database, model.
# Usage: ./scripts/setup.sh [--skip-model]
set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_MODEL=0
[[ "${1:-}" == "--skip-model" ]] && SKIP_MODEL=1

# 1. Find a supported Python (3.10 - 3.13). System Python is never modified.
PY=""
for c in python3.12 python3.11 python3.13 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)'; then
    PY="$c"; break
  fi
done
if [[ -z "$PY" ]]; then
  echo "ERROR: Python 3.10-3.13 not found. Install one, e.g.: brew install python@3.12  (or from python.org)" >&2
  exit 1
fi
echo "Using $($PY --version) at $(command -v $PY)"

# 2. Virtual environment inside the project
if [[ ! -d .venv ]]; then
  "$PY" -m venv .venv || { echo "ERROR: could not create .venv" >&2; exit 1; }
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
echo "Installing dependencies (first run downloads PyTorch, ~1 GB; this can take several minutes)..."
pip install -r requirements.txt || { echo "ERROR: dependency installation failed. Check your internet connection and re-run." >&2; exit 1; }

# 3. Local directories + database
mkdir -p data models reports
python - <<'PY'
from config.settings import DB_PATH, ensure_directories
from database.db_manager import DatabaseManager
ensure_directories(); DatabaseManager(DB_PATH); print(f"Database ready: {DB_PATH}")
PY

# 4. Model (only if missing)
if [[ $SKIP_MODEL -eq 0 ]]; then
  python scripts/download_model.py || echo "WARNING: model download failed - simulation-only mode will still work."
fi

echo
echo "Setup complete. Verify with:  source .venv/bin/activate && pytest"
echo "Launch with:                  ./scripts/run.sh"
