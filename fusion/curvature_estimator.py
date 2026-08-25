"""
Curvature estimator for smart-adaptive-headlight.

Computes road curvature radius R and hill-crest/slope flags from
lane polynomials and optional depth-map vertical gradients.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass
class CurvatureInfo:
    """Road curvature and slope information."""
    radius: float           # metres (large = straight)
    direction: str          # "left", "right", or "straight"
    is_hill_crest: bool     # True if approaching a hill crest
    slope_deg: float        # Estimated road slope in degrees (positive = uphill)


class CurvatureEstimator:
    """Estimates road curvature and slope."""

    def __init__(
        self,
        ym_per_pix: float = 30.0 / 720,
        xm_per_pix: float = 3.7 / 680,
        hill_crest_thresh: float = 0.15,
    ) -> None:
        self._ym = ym_per_pix
        self._xm = xm_per_pix
        self._hill_thresh = hill_crest_thresh

    def estimate(
        self,
        lane_poly: Optional[np.ndarray],
        frame_height: int = 720,
        depth_map: Optional[np.ndarray] = None,
    ) -> CurvatureInfo:
        """Compute curvature from lane polynomial and optional depth map.

        Args:
            lane_poly: 2nd-degree polynomial coefficients [a, b, c].
            frame_height: Image height for evaluation point.
            depth_map: Optional depth map for slope estimation.
        """
        # Curvature from lane polynomial
        radius = 9999.0
        direction = "straight"

        if lane_poly is not None and len(lane_poly) >= 3:
            a, b, _ = lane_poly[:3]
            y_eval = frame_height * self._ym
            a_m = a * self._xm / (self._ym ** 2)
            b_m = b * self._xm / self._ym
            denom = abs(2 * a_m)
            if denom > 1e-8:
                radius = ((1 + (2 * a_m * y_eval + b_m) ** 2) ** 1.5) / denom
                radius = float(np.clip(radius, 1.0, 99999.0))

            # Direction based on sign of curvature coefficient
            if a_m > 1e-6:
                direction = "right"
            elif a_m < -1e-6:
                direction = "left"

        # Hill crest / slope from depth map vertical gradient
        is_hill_crest = False
        slope_deg = 0.0

        if depth_map is not None:
            h = depth_map.shape[0]
            # Sample vertical strip at center
            center_col = depth_map.shape[1] // 2
            strip_w = 50
            col_start = max(0, center_col - strip_w)
            col_end = min(depth_map.shape[1], center_col + strip_w)
            vertical_profile = depth_map[:, col_start:col_end].mean(axis=1)

            # Compute gradient of depth along vertical axis
            grad = np.gradient(vertical_profile)
            upper_grad = grad[:h // 3].mean() if h > 3 else 0
            lower_grad = grad[2 * h // 3:].mean() if h > 3 else 0

            # Hill crest: depth increases then decreases (sign change)
            if upper_grad > self._hill_thresh and lower_grad < -self._hill_thresh:
                is_hill_crest = True

            # Slope estimate from average gradient
            avg_grad = float(np.mean(grad[h // 4: 3 * h // 4]))
            slope_deg = float(np.degrees(np.arctan(avg_grad * 0.1)))
            slope_deg = np.clip(slope_deg, -30.0, 30.0)

        return CurvatureInfo(
            radius=radius, direction=direction,
            is_hill_crest=is_hill_crest, slope_deg=slope_deg,
        )


if __name__ == "__main__":
    ce = CurvatureEstimator()
    # Gentle right curve
    info = ce.estimate(np.array([2e-4, -0.3, 400.0]))
    print(f"Curve: R={info.radius:.0f}m, dir={info.direction}, "
          f"hill={info.is_hill_crest}, slope={info.slope_deg:.1f}°")
    # Straight road
    info2 = ce.estimate(np.array([0.0, 0.0, 640.0]))
    print(f"Straight: R={info2.radius:.0f}m, dir={info2.direction}")
    # No lane data
    info3 = ce.estimate(None)
    print(f"No data: R={info3.radius:.0f}m, dir={info3.direction}")
