"""
Object classifier — lane-role assignment for tracked objects.

Given a track, lane polynomials, and relative motion, classify as:
  ONCOMING | SAME_LANE_AHEAD | ADJACENT_LANE | PARKED | IRRELEVANT
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class LaneRole(Enum):
    ONCOMING = "oncoming"
    SAME_LANE_AHEAD = "same_lane_ahead"
    ADJACENT_LANE = "adjacent_lane"
    PARKED = "parked"
    IRRELEVANT = "irrelevant"


class ObjectClassifier:
    """Classifies each tracked object's lane-role for glare risk assessment."""

    def __init__(
        self,
        lane_width_px: float = 200.0,
        parked_speed_thresh: float = 2.0,
        frame_width: int = 1280,
    ) -> None:
        """
        Args:
            lane_width_px: Estimated lane width in pixels (at bottom of frame).
            parked_speed_thresh: Pixel velocity below which an object is 'parked'.
            frame_width: Image width for center reference.
        """
        self._lane_w = lane_width_px
        self._parked_thresh = parked_speed_thresh
        self._frame_cx = frame_width // 2

    def classify(
        self,
        track_center: tuple[int, int],
        track_velocity: tuple[float, float],
        lane_poly: Optional[np.ndarray],
        adjacent_poly: Optional[np.ndarray],
        frame_height: int = 720,
    ) -> LaneRole:
        """Classify a single tracked object's lane role.

        Args:
            track_center: (cx, cy) in pixels.
            track_velocity: (vx, vy) in px/frame.
            lane_poly: Left lane polynomial (deg 2) or None.
            adjacent_poly: Right lane polynomial (deg 2) or None.
            frame_height: Image height.

        Returns:
            LaneRole enum value.
        """
        cx, cy = track_center
        vx, vy = track_velocity
        speed = np.sqrt(vx ** 2 + vy ** 2)

        # If nearly stationary → parked
        if speed < self._parked_thresh:
            return LaneRole.PARKED

        # Determine lateral offset from ego lane center
        if lane_poly is not None and adjacent_poly is not None:
            lane_x = np.polyval(lane_poly, cy)
            adj_x = np.polyval(adjacent_poly, cy)
            lane_center = (lane_x + adj_x) / 2.0
            offset = cx - lane_center
        else:
            # Fallback: use frame center as lane center
            offset = cx - self._frame_cx

        # Moving toward ego vehicle (vy > 0 means moving down = approaching)
        # In bird's-eye, oncoming traffic appears to move downward
        is_approaching = vy > 1.0

        # Lateral position classification
        half_lane = self._lane_w / 2.0

        if abs(offset) < half_lane * 0.6:
            # Within ego lane
            if is_approaching:
                return LaneRole.ONCOMING
            else:
                return LaneRole.SAME_LANE_AHEAD
        elif abs(offset) < half_lane * 1.5:
            # Adjacent lane
            if offset < 0 and is_approaching:
                return LaneRole.ONCOMING  # Oncoming in adjacent lane
            return LaneRole.ADJACENT_LANE
        else:
            return LaneRole.IRRELEVANT


if __name__ == "__main__":
    oc = ObjectClassifier()
    # Test cases
    tests = [
        ((640, 500), (0.5, 5.0), None, None, "Approaching center"),
        ((640, 500), (0.0, -3.0), None, None, "Moving away center"),
        ((300, 500), (0.0, 4.0), None, None, "Approaching left"),
        ((640, 500), (0.1, 0.5), None, None, "Slow center"),
        ((100, 500), (0.0, 0.0), None, None, "Stationary far left"),
    ]
    for center, vel, lp, ap, desc in tests:
        role = oc.classify(center, vel, lp, ap)
        print(f"  {desc:30s} → {role.value}")
