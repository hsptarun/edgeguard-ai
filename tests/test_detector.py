"""Detector tests. Pipeline tests that need the model are skipped when weights are not downloaded."""
import cv2
import numpy as np
import pytest

from detection.person_detector import (Detection, ModelNotAvailableError, PersonDetector, VideoError, annotate_frame,
                                       available_devices, iter_frames, open_video, video_info)
from config.settings import MODEL_PATH


def _make_video(path, n=12):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for i in range(n):
        w.write(np.full((120, 160, 3), i * 10, np.uint8))
    w.release()


def test_missing_model_is_reported(tmp_path):
    det = PersonDetector(model_path=tmp_path / "nope.pt")
    assert not det.is_ready()
    with pytest.raises(ModelNotAvailableError):
        det.detect(np.zeros((10, 10, 3), np.uint8))


def test_video_validation(tmp_path):
    with pytest.raises(VideoError):
        open_video(tmp_path / "x.txt")
    with pytest.raises(VideoError):
        open_video(tmp_path / "missing.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video" * 100)
    with pytest.raises(VideoError):
        cap = open_video(bad)
        list(iter_frames(cap))


def test_frame_iteration_with_stride(tmp_path):
    p = tmp_path / "v.mp4"
    _make_video(p)
    cap = open_video(p)
    assert video_info(cap)["width"] == 160
    idx = [i for i, _, _ in iter_frames(cap, stride=5)]
    cap.release()
    assert idx[:2] == [0, 5]
    with pytest.raises(ValueError):
        list(iter_frames(open_video(p), stride=0))


def test_annotate_does_not_mutate_input():
    f = np.zeros((100, 100, 3), np.uint8)
    out = annotate_frame(f, [Detection(10, 10, 50, 80, 0.9)], (0.1, 0.1, 0.9, 0.9), [True], "Zone")
    assert f.sum() == 0 and out.sum() > 0


def test_foot_point_and_devices():
    assert Detection(0, 0, 10, 20, 0.5).foot_point == (5, 20)
    assert "cpu" in available_devices()


def _inference_stack_available() -> bool:
    try:
        import torch  # noqa: F401
        import ultralytics  # noqa: F401
    except Exception:
        return False
    return MODEL_PATH.exists()


@pytest.mark.skipif(not _inference_stack_available(), reason="model weights or torch/ultralytics not available")
def test_real_model_runs_on_blank_frame():
    dets, ms = PersonDetector().detect(np.zeros((480, 640, 3), np.uint8), 0.4)
    assert dets == [] and ms > 0
    with pytest.raises(ValueError):
        PersonDetector().detect(np.zeros((480, 640, 3), np.uint8), 1.5)
