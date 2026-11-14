"""Project-wide paths and constants. All paths are relative to the project root."""
from __future__ import annotations

import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIG_DIR = PROJECT_ROOT / "config"

DB_PATH = DATA_DIR / "edgeguard.db"
BUILDING_CONFIG_PATH = CONFIG_DIR / "building_config.json"

# Pretrained person detector (COCO class 0 = "person").
# License: Ultralytics YOLO models are AGPL-3.0 (https://github.com/ultralytics/ultralytics).
MODEL_NAME = "yolov8n.pt"
MODEL_PATH = MODELS_DIR / MODEL_NAME
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
MODEL_LICENSE = "AGPL-3.0 (Ultralytics YOLOv8n, pretrained on COCO)"
MIN_MODEL_BYTES = 1_000_000  # sanity check for truncated downloads

ALLOWED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
MAX_UPLOAD_MB = 200
DEFAULT_CONFIDENCE = 0.40


def ensure_directories() -> None:
    """Create generated-data directories if they do not exist."""
    for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure console logging once (safe to call repeatedly)."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
