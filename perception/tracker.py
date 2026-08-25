"""
Multi-object tracker for smart-adaptive-headlight.

Lightweight IOU + Kalman tracker — no ByteTrack dependency required.
Maintains persistent track IDs and estimates pixel velocity.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from diagnostics.health_monitor import get_monitor as _get_monitor
    _HAS_DIAGNOSTICS = True
except ImportError:
    _HAS_DIAGNOSTICS = False


@dataclass
class Track:
    """A tracked object with persistent ID and velocity estimate."""
    track_id: int
    bbox: tuple[int, int, int, int]
    cls: int
    cls_name: str
    conf: float
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    velocity: tuple[float, float] = (0.0, 0.0)  # px/frame (dx, dy)
    center: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _KalmanBoxState:
    """Minimal 2D center + size Kalman filter."""

    def __init__(self, bbox: tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        # state: [cx, cy, w, h, vx, vy]
        self.x = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(6) * 100.0
        self.F = np.eye(6); self.F[0, 4] = 1; self.F[1, 5] = 1
        self.H = np.eye(4, 6)
        self.Q = np.eye(6) * 1.0; self.Q[4, 4] = 0.01; self.Q[5, 5] = 0.01
        self.R = np.eye(4) * 10.0

    def predict(self) -> tuple[int, int, int, int]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._to_bbox()

    def update(self, bbox: tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = bbox
        z = np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def _to_bbox(self) -> tuple[int, int, int, int]:
        cx, cy, w, h = self.x[:4]
        return (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))

    @property
    def velocity(self) -> tuple[float, float]:
        return (float(self.x[4]), float(self.x[5]))


class Tracker:
    """IOU + Kalman multi-object tracker."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10) -> None:
        self._iou_thresh = iou_threshold
        self._max_age = max_age
        self._next_id = 1
        self._tracks: list[tuple[Track, _KalmanBoxState]] = []

    def update(self, detections: list, frame: np.ndarray | None = None) -> List[Track]:
        """Match detections to existing tracks and return updated track list.

        Args:
            detections: List of Detection objects (need .bbox, .cls, .cls_name, .conf).
            frame: Optional image frame for visual processing.

        Returns:
            Active tracks with updated positions and velocities.
        """
        # Predict all existing tracks
        for trk, kal in self._tracks:
            kal.predict()

        # Build IOU cost matrix
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(self._tracks)))
        matches: list[tuple[int, int]] = []

        if detections and self._tracks:
            iou_matrix = np.zeros((len(detections), len(self._tracks)))
            for di, det in enumerate(detections):
                for ti, (trk, kal) in enumerate(self._tracks):
                    iou_matrix[di, ti] = _iou(det.bbox, kal._to_bbox())

            # Greedy matching
            while True:
                if iou_matrix.size == 0:
                    break
                best = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                if iou_matrix[best] < self._iou_thresh:
                    break
                di, ti = int(best[0]), int(best[1])
                matches.append((di, ti))
                iou_matrix[di, :] = 0; iou_matrix[:, ti] = 0
                unmatched_dets.remove(di); unmatched_trks.remove(ti)

        # Update matched tracks
        for di, ti in matches:
            det = detections[di]
            trk, kal = self._tracks[ti]
            kal.update(det.bbox)
            trk.bbox = det.bbox
            trk.conf = det.conf
            trk.hits += 1
            trk.time_since_update = 0
            trk.velocity = kal.velocity
            trk.__post_init__()

        # Create new tracks for unmatched detections
        for di in unmatched_dets:
            det = detections[di]
            kal = _KalmanBoxState(det.bbox)
            trk = Track(
                track_id=self._next_id, bbox=det.bbox,
                cls=det.cls, cls_name=det.cls_name, conf=det.conf,
            )
            self._next_id += 1
            self._tracks.append((trk, kal))

        # Age unmatched tracks
        for ti in unmatched_trks:
            self._tracks[ti][0].time_since_update += 1

        # Increment age, remove dead tracks
        alive = []
        for trk, kal in self._tracks:
            trk.age += 1
            if trk.time_since_update <= self._max_age:
                alive.append((trk, kal))
        self._tracks = alive

        active = [trk for trk, _ in self._tracks if trk.time_since_update == 0]

        # --- Health diagnostics instrumentation ---
        if _HAS_DIAGNOSTICS:
            n_new = len(unmatched_dets)
            n_det = max(len(detections), 1)
            _get_monitor().record(
                "tracker", {"n_new_ids": n_new, "n_detections": n_det}
            )

        return active


# Alias for backward compatibility and test consistency
ObjectTracker = Tracker


if __name__ == "__main__":
    from perception.camera_capture import Camera
    from perception.detector import Detector
    import cv2

    cam = Camera(mock=True)
    det = Detector(mock=True)
    tracker = Tracker()
    print("[Tracker] Running mock demo. Press 'q' to quit.")
    for frame in cam.stream():
        dets = det.detect(frame)
        tracks = tracker.update(dets)
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"ID:{t.track_id} v=({t.velocity[0]:.0f},{t.velocity[1]:.0f})",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.imshow("Tracker", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()
