"""Classification and latency metrics implemented with plain Python (no scikit-learn dependency).

Convention: "positive" = suspicious. When a ratio's denominator is zero the metric is reported as 0.0.
"""
from __future__ import annotations

import statistics
from typing import Sequence


def confusion_matrix(y_true: Sequence[bool], y_pred: Sequence[bool]) -> dict[str, int]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def classification_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred)
    precision = _div(cm["tp"], cm["tp"] + cm["fp"])
    recall = _div(cm["tp"], cm["tp"] + cm["fn"])
    return {
        **cm,
        "n": len(y_true),
        "accuracy": _div(cm["tp"] + cm["tn"], len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": _div(2 * precision * recall, precision + recall),
        "false_positive_rate": _div(cm["fp"], cm["fp"] + cm["tn"]),
    }


def latency_summary(ms: Sequence[float], wall_seconds: float | None = None, frames: int | None = None) -> dict[str, float]:
    """Summarise per-frame inference times (milliseconds)."""
    if not ms:
        return {"frames_timed": 0}
    ordered = sorted(ms)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    out = {
        "frames_timed": len(ms),
        "inference_ms_mean": statistics.fmean(ms),
        "inference_ms_median": statistics.median(ms),
        "inference_ms_p95": p95,
        "inference_fps": 1000.0 / statistics.fmean(ms),
    }
    if wall_seconds and frames:
        out["pipeline_fps_wall_clock"] = frames / wall_seconds
    return out
