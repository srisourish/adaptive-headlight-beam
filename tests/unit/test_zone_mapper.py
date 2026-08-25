"""Unit test for Zone Mapper (decision.zone_mapper)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from decision.zone_mapper import ZoneMapper


def test_zone_mapper_pixel_mapping() -> None:
    """Test object x-coordinate pixel center to zone index conversion."""
    zm = ZoneMapper()
    # Left edge (x=50 in 1280w frame) -> Zone 0
    zone_left = zm.object_to_zone(center_x=50, frame_width=1280)
    assert zone_left == 0

    # Center (x=640 in 1280w frame) -> Middle zone (3 or 4 for 8 zones)
    zone_center = zm.object_to_zone(center_x=640, frame_width=1280)
    assert zone_center in (3, 4)

    # Right edge (x=1200 in 1280w frame) -> Zone 7
    zone_right = zm.object_to_zone(center_x=1200, frame_width=1280)
    assert zone_right == zm.zone_count - 1


def test_zone_brightness_gamma_calculation() -> None:
    """Test gamma power mapping: zero risk -> max power, high risk -> min power."""
    zm = ZoneMapper()

    # Zero risk across all zones
    brightness_max = zm.compute_brightness([0.0] * 8)
    for b in brightness_max:
        assert b == 255

    # 100% risk across all zones
    brightness_min = zm.compute_brightness([100.0] * 8)
    for b in brightness_min:
        assert b == 10  # min brightness floor
