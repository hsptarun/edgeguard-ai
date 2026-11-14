"""Local person detection with a pretrained Ultralytics YOLO model (COCO class 0 = person).

Inference runs entirely on this machine. The only network access is the one-time model download.
The detector answers "is a person-shaped object visible here?" - nothing about identity or intent.
"""
from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from config.settings import ALLOWED_VIDEO_EXTENSIONS, MIN_MODEL_BYTES, MODEL_PATH, MODEL_URL

log = logging.getLogger(__name__)


class ModelNotAvailableError(RuntimeError):
    """Model weights or the ultralytics package are missing."""


class VideoError(ValueError):
    """Video file is unsupported, missing or corrupted."""


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre of the box - a better 'where is the person standing' proxy than the centre."""
        return ((self.x1 + self.x2) / 2, self.y2)


def download_model(dest: Path = MODEL_PATH, url: str = MODEL_URL) -> Path:
    """Download weights once (needs internet). Writes to a temp file first, then renames."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    try:
        log.info("Downloading model from %s", url)
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
        if tmp.stat().st_size < MIN_MODEL_BYTES:
            raise OSError(f"Downloaded file is too small ({tmp.stat().st_size} bytes); download incomplete.")
        tmp.replace(dest)
    except OSError as exc:
        if tmp.exists():
            tmp.unlink()
        raise ModelNotAvailableError(f"Model download failed: {exc}. Check your internet connection.") from exc
    return dest


def available_devices() -> list[str]:
    """'cpu' always; 'mps' (Apple GPU) only if the installed PyTorch reports it as available."""
    devices = ["cpu"]
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            devices.append("mps")
    except Exception:  # torch missing or incompatible
        pass
    return devices


class PersonDetector:
    def __init__(self, model_path: Path = MODEL_PATH, device: str = "cpu", imgsz: int = 640):
        self.model_path, self.device, self.imgsz = Path(model_path), device, imgsz
        self._model = None

    def is_ready(self) -> bool:
        return self.model_path.exists() and self.model_path.stat().st_size >= MIN_MODEL_BYTES

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.is_ready():
            raise ModelNotAvailableError(f"Model file not found at {self.model_path}. Run scripts/download_model.py.")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelNotAvailableError("The 'ultralytics' package is not installed.") from exc
        try:
            self._model = YOLO(str(self.model_path))
        except Exception as exc:  # corrupted weights etc.
            raise ModelNotAvailableError(f"Could not load model: {exc}") from exc
        log.info("Loaded %s on %s", self.model_path.name, self.device)

    def detect(self, frame: np.ndarray, confidence: float = 0.4) -> tuple[list[Detection], float]:
        """Return (person detections, inference milliseconds) for one BGR frame."""
        if frame is None or frame.ndim != 3:
            raise ValueError("frame must be a BGR image array (H, W, 3).")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1 (exclusive).")
        self.load()
        t0 = time.perf_counter()
        res = self._model.predict(frame, classes=[0], conf=confidence, imgsz=self.imgsz,
                                  device=self.device, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        boxes = res.boxes
        dets = [Detection(*map(float, xyxy), float(c))
                for xyxy, c in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy())]
        return dets, ms


# ---- video helpers ---------------------------------------------------------------------
def open_video(path: str | Path) -> cv2.VideoCapture:
    path = Path(path)
    if path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise VideoError(f"Unsupported file type '{path.suffix}'. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}.")
    if not path.exists():
        raise VideoError("Video file not found.")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoError("The video could not be opened. The file may be corrupted or use an unsupported codec.")
    return cap


def video_info(cap: cv2.VideoCapture) -> dict:
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    return {"fps": fps if fps > 0 else 25.0, "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}


def iter_frames(cap: cv2.VideoCapture, stride: int = 1, max_frames: int | None = None) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, stream_seconds, frame) for every `stride`-th frame."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    fps = video_info(cap)["fps"]
    idx = yielded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            if idx == 0:
                raise VideoError("No frames could be decoded; the video appears to be corrupted.")
            return
        if idx % stride == 0:
            yield idx, idx / fps, frame
            yielded += 1
            if max_frames and yielded >= max_frames:
                return
        idx += 1


def annotate_frame(frame: np.ndarray, dets: list[Detection], zone: tuple[float, float, float, float] | None = None,
                   inside: list[bool] | None = None, zone_label: str = "") -> np.ndarray:
    """Draw boxes (red = inside zone, green = outside) and the monitored zone rectangle."""
    out = frame.copy()
    h, w = out.shape[:2]
    if zone:
        x1, y1, x2, y2 = zone
        cv2.rectangle(out, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 165, 255), 2)
        if zone_label:
            cv2.putText(out, zone_label, (int(x1 * w) + 6, int(y1 * h) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    for i, d in enumerate(dets):
        color = (0, 0, 255) if inside and inside[i] else (0, 200, 0)
        cv2.rectangle(out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, 2)
        cv2.putText(out, f"person {d.confidence:.2f}", (int(d.x1), max(14, int(d.y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out
