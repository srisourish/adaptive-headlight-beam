"""
Object detector for smart-adaptive-headlight.

Wraps Ultralytics YOLO (default: yolov8n) with a filtered class set
relevant to headlight/glare risk assessment.  Falls back to a
lightweight mock detector when the model weights are unavailable.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config

try:
    from diagnostics.health_monitor import get_monitor as _get_monitor
    _HAS_DIAGNOSTICS = True
except ImportError:
    _HAS_DIAGNOSTICS = False

# COCO class names relevant to this project
_RELEVANT_CLASSES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class Detection:
    """Single detection result."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    cls: int  # COCO class ID
    cls_name: str  # human-readable class name
    conf: float  # confidence score [0, 1]
    center: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)


class Detector:
    """YOLO-based object detector with mock fallback."""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf_threshold: float | None = None,
        mock: bool = False,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            weights: Path or Ultralytics model name (e.g. 'yolov8n.pt').
            conf_threshold: Minimum confidence. If None, read from config.
            mock: Force mock mode (random detections).
            device: 'cpu', 'cuda', or 'cuda:0'.
        """
        thresholds = get_config("thresholds")
        self._conf_thresh = conf_threshold or thresholds.get("detection", {}).get(
            "min_confidence", 0.35
        )
        relevant = thresholds.get("detection", {}).get("relevant_classes", list(_RELEVANT_CLASSES.keys()))
        self._relevant_ids = set(relevant)
        self._mock = mock
        self._model = None

        if not mock:
            try:
                from ultralytics import YOLO

                self._model = YOLO(weights)
                self._model.to(device)
            except Exception as exc:
                print(
                    f"[Detector] Could not load YOLO model ({exc}); "
                    "falling back to mock mode."
                )
                self._mock = True

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single frame.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            List of ``Detection`` objects for relevant classes above threshold.
        """
        if self._mock:
            detections = self._mock_detect(frame)
        else:
            detections = self._real_detect(frame)

        # --- Health diagnostics instrumentation ---
        if _HAS_DIAGNOSTICS:
            n = len(detections)
            avg_conf = (sum(d.conf for d in detections) / n) if n > 0 else 0.0
            _get_monitor().record("detector", {"conf": avg_conf, "n_detections": n})

        return detections

    def _real_detect(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO inference on a real frame."""
        results = self._model(frame, verbose=False)  # type: ignore[misc]
        detections: list[Detection] = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id not in self._relevant_ids:
                    continue
                if conf < self._conf_thresh:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        cls=cls_id,
                        cls_name=_RELEVANT_CLASSES.get(cls_id, f"class_{cls_id}"),
                        conf=conf,
                    )
                )
        return detections

    def _mock_detect(self, frame: np.ndarray) -> List[Detection]:
        """Generate synthetic detections from the mock frame."""
        h, w = frame.shape[:2]
        detections: list[Detection] = []

        # Simple brightness-blob detection as a proxy
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Heuristic: tall → person, wide → vehicle
            aspect = bh / max(bw, 1)
            if aspect > 1.5:
                cls_id, cls_name = 0, "person"
            else:
                cls_id, cls_name = 2, "car"
            detections.append(
                Detection(
                    bbox=(x, y, x + bw, y + bh),
                    cls=cls_id,
                    cls_name=cls_name,
                    conf=0.75 + 0.2 * np.random.random(),
                )
            )

        if not detections:
            # Fallback default synthetic vehicle detection for blank test frames
            detections.append(
                Detection(
                    bbox=(int(w * 0.35), int(h * 0.5), int(w * 0.42), int(h * 0.6)),
                    cls=2,
                    cls_name="car",
                    conf=0.92,
                )
            )

        return detections


if __name__ == "__main__":
    import argparse

    from perception.camera_capture import Camera

    parser = argparse.ArgumentParser(description="Detector demo")
    parser.add_argument("--source", default=None)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    cam = Camera(source=args.source, mock=args.mock or args.source is None)
    det = Detector(mock=args.mock or args.source is None)
    print(f"[Detector] Mock={det._mock}. Press 'q' to quit.")

    for frame in cam.stream():
        dets = det.detect(frame)
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            color = (0, 255, 0) if d.cls == 0 else (255, 100, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{d.cls_name} {d.conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        cv2.imshow("Detections", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
