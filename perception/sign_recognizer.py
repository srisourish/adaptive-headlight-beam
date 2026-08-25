"""
Traffic sign recognizer for smart-adaptive-headlight.

Wraps a YOLO-tiny model scoped to traffic sign classes. Falls back
to a stub that returns empty results when no model is available.
Interface: recognize(frame) -> List[Sign]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Relevant sign classes for headlight adaptation
SIGN_CLASSES = {
    0: "speed_limit",
    1: "no_overtaking",
    2: "construction",
    3: "stop",
    4: "yield",
    5: "no_entry",
    6: "highway_begin",
    7: "highway_end",
    8: "curve_warning",
}


@dataclass
class Sign:
    """Detected traffic sign."""
    bbox: tuple[int, int, int, int]
    sign_class: str
    conf: float
    value: Optional[str] = None  # e.g., "60" for speed_limit


class SignRecognizer:
    """Traffic sign recognition with YOLO-tiny and mock fallback."""

    def __init__(self, weights: Optional[str] = None, mock: bool = False) -> None:
        self._mock = mock
        self._model = None

        if weights and not mock:
            try:
                from ultralytics import YOLO
                self._model = YOLO(weights)
            except Exception as exc:
                print(f"[SignRecognizer] Could not load model ({exc}); using mock.")
                self._mock = True
        else:
            self._mock = True

    def recognize(self, frame: np.ndarray) -> List[Sign]:
        """Detect and classify traffic signs in a frame.

        Args:
            frame: BGR image.

        Returns:
            List of Sign objects.
        """
        if self._mock:
            return self._mock_recognize(frame)

        results = self._model(frame, verbose=False)
        signs: list[Sign] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.4:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_name = SIGN_CLASSES.get(cls_id, f"sign_{cls_id}")
                signs.append(Sign(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    sign_class=cls_name,
                    conf=conf,
                ))
        return signs

    def _mock_recognize(self, frame: np.ndarray) -> List[Sign]:
        """Return empty list — no signs detected in mock mode."""
        # In a real scenario with GTSRB-trained model, this would detect signs.
        # For mock mode, we return empty to avoid false positives.
        return []


if __name__ == "__main__":
    from perception.camera_capture import Camera
    cam = Camera(mock=True)
    sr = SignRecognizer(mock=True)
    print("[SignRecognizer] Mock demo (no detections). Press 'q' to quit.")
    for frame in cam.stream():
        signs = sr.recognize(frame)
        for s in signs:
            x1, y1, x2, y2 = s.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, s.sign_class, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, f"Signs: {len(signs)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Signs", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()
