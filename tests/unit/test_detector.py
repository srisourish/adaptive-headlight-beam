"""Unit test for Object Detector (perception.detector)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from perception.detector import Detection, Detector


def test_detector_mock_mode() -> None:
    """Test Detector in mock mode returns valid detections on synthetic frame."""
    detector = Detector(mock=True)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    detections = detector.detect(frame)
    assert isinstance(detections, list)
    assert len(detections) > 0, "Mock detector should return at least one mock detection"

    for det in detections:
        assert isinstance(det, Detection)
        assert len(det.bbox) == 4
        x1, y1, x2, y2 = det.bbox
        assert 0 <= x1 < x2 <= 1280
        assert 0 <= y1 < y2 <= 720
        assert 0.0 <= det.conf <= 1.0
        assert isinstance(det.cls_name, str)
        assert det.center == ((x1 + x2) // 2, (y1 + y2) // 2)


def test_detection_dataclass() -> None:
    """Test Detection dataclass post-init center calculation."""
    det = Detection(bbox=(100, 200, 300, 400), cls=2, cls_name="car", conf=0.95)
    assert det.center == (200, 300)
