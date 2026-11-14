#!/usr/bin/env python3
"""Download the pretrained YOLOv8n weights once (requires internet). Safe to re-run."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import MODEL_LICENSE, MODEL_PATH, MODEL_URL, MIN_MODEL_BYTES, ensure_directories  # noqa: E402
from detection.person_detector import ModelNotAvailableError, download_model  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ensure_directories()
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size >= MIN_MODEL_BYTES:
        print(f"Model already present: {MODEL_PATH}")
        return 0
    print(f"Source : {MODEL_URL}\nLicense: {MODEL_LICENSE}")
    try:
        path = download_model()
    except ModelNotAvailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("The dashboard still works in simulation-only mode. Re-run this script when online.", file=sys.stderr)
        return 1
    print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
