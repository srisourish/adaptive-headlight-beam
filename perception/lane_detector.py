"""
Classical lane detector for smart-adaptive-headlight.

Uses Canny edge detection + Hough line transform + polynomial fitting
in a bird's-eye-view warp. Returns lane polynomials and curvature radius.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config


@dataclass
class LaneResult:
    """Result of lane detection for a single frame."""
    lane_poly: Optional[np.ndarray]       # Left lane polynomial coeffs (deg 2)
    adjacent_poly: Optional[np.ndarray]   # Right lane polynomial coeffs (deg 2)
    curvature_radius: float               # Estimated radius of curvature (metres)
    bev_image: Optional[np.ndarray] = None  # Bird's-eye debug image


class LaneDetector:
    """Classical CV lane detector with BEV warp."""

    def __init__(self, mock: bool = False) -> None:
        self._mock = mock
        calib = get_config("camera_calib")
        src = calib.get("bev_src_points", [[200,720],[580,450],[700,450],[1100,720]])
        dst = calib.get("bev_dst_points", [[300,720],[300,0],[980,0],[980,720]])
        self._src = np.float32(src)
        self._dst = np.float32(dst)
        self._M = cv2.getPerspectiveTransform(self._src, self._dst)
        self._Minv = cv2.getPerspectiveTransform(self._dst, self._src)
        self._ym_per_pix = 30.0 / 720  # metres per pixel in y (approx)
        self._xm_per_pix = 3.7 / 680   # metres per pixel in x (approx)

    def detect(self, frame: np.ndarray) -> LaneResult:
        if self._mock:
            return self._mock_lanes(frame)
        h, w = frame.shape[:2]
        # Warp to bird's-eye view
        bev = cv2.warpPerspective(frame, self._M, (w, h))
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Split left / right halves
        mid = w // 2
        left_edge = edges[:, :mid]
        right_edge = edges[:, mid:]

        left_poly = self._fit_lane(left_edge, offset=0)
        right_poly = self._fit_lane(right_edge, offset=mid)

        # Compute curvature from left lane (or right if left unavailable)
        curv = self._curvature(left_poly if left_poly is not None else right_poly, h)

        return LaneResult(
            lane_poly=left_poly,
            adjacent_poly=right_poly,
            curvature_radius=curv,
            bev_image=bev,
        )

    def _fit_lane(self, edge_img: np.ndarray, offset: int) -> Optional[np.ndarray]:
        ys, xs = np.nonzero(edge_img)
        if len(xs) < 50:
            return None
        xs = xs + offset
        try:
            poly = np.polyfit(ys, xs, 2)
            return poly
        except np.RankWarning:
            return None

    def _curvature(self, poly: Optional[np.ndarray], img_h: int) -> float:
        if poly is None:
            return 9999.0  # Effectively straight
        a, b, _ = poly
        y_eval = img_h * self._ym_per_pix
        a_m = a * self._xm_per_pix / (self._ym_per_pix ** 2)
        b_m = b * self._xm_per_pix / self._ym_per_pix
        denom = abs(2 * a_m)
        if denom < 1e-6:
            return 9999.0
        R = ((1 + (2 * a_m * y_eval + b_m) ** 2) ** 1.5) / denom
        return float(np.clip(R, 1.0, 99999.0))

    def _mock_lanes(self, frame: np.ndarray) -> LaneResult:
        h, w = frame.shape[:2]
        left_poly = np.array([1e-4, -0.3, w * 0.35])
        right_poly = np.array([1e-4, -0.3, w * 0.65])
        curv = self._curvature(left_poly, h)
        return LaneResult(lane_poly=left_poly, adjacent_poly=right_poly,
                          curvature_radius=curv, bev_image=None)


if __name__ == "__main__":
    from perception.camera_capture import Camera
    cam = Camera(mock=True)
    ld = LaneDetector(mock=True)
    print("[LaneDetector] Mock demo. Press 'q' to quit.")
    for frame in cam.stream():
        result = ld.detect(frame)
        info = f"R={result.curvature_radius:.0f}m"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Lane Detection", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()
