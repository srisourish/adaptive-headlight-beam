"""
Zone mapper for smart-adaptive-headlight.

Maps per-object glare risk scores to N angular beam zones.
Computes zone brightness using: brightness = B_max * (1 - (risk/100)^gamma)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config


class ZoneMapper:
    """Maps object-level risk to angular zone brightness commands."""

    def __init__(self) -> None:
        cfg = get_config("zones")
        self._n_zones = cfg.get("zone_count", 8)
        self._boundaries = cfg.get(
            "zone_boundaries", list(np.linspace(-40, 40, self._n_zones + 1))
        )
        self._gamma = cfg.get("gamma", 1.5)
        self._b_min = cfg.get("brightness_min", 10)
        self._b_max = cfg.get("brightness_max", 255)
        self._frame_width = get_config("camera_calib").get("frame_width", 1280)

    @property
    def zone_count(self) -> int:
        return self._n_zones

    def object_to_zone(
        self, center_x: int, frame_width: int | None = None
    ) -> int:
        """Map an object's horizontal pixel center to a zone index.

        Args:
            center_x: Object center x-coordinate in pixels.
            frame_width: Image width (uses config default if None).

        Returns:
            Zone index [0, zone_count - 1].
        """
        w = frame_width or self._frame_width
        # Convert pixel to angle (assuming linear FOV mapping)
        fov_total = self._boundaries[-1] - self._boundaries[0]
        angle = self._boundaries[0] + (center_x / w) * fov_total

        for i in range(self._n_zones):
            if self._boundaries[i] <= angle < self._boundaries[i + 1]:
                return i
        # Edge case: rightmost boundary
        return self._n_zones - 1

    def aggregate_risks(
        self,
        object_risks: list[tuple[int, float]],
    ) -> list[float]:
        """Aggregate per-object risks into per-zone risk (max per zone).

        Args:
            object_risks: List of (zone_index, risk_score) pairs.

        Returns:
            Per-zone risk array (length = zone_count), each [0, 100].
        """
        zone_risks = [0.0] * self._n_zones
        for pos, risk in object_risks:
            zone_idx = pos if 0 <= pos < self._n_zones else self.object_to_zone(pos)
            if 0 <= zone_idx < self._n_zones:
                zone_risks[zone_idx] = max(zone_risks[zone_idx], risk)
        return zone_risks

    def brightness(self, risk: float) -> int:
        """Compute zone brightness from risk score.

        Formula: B = B_max * (1 - (risk / 100) ^ gamma)
        Clamped to [B_min, B_max].
        """
        normalised = np.clip(risk / 100.0, 0, 1)
        b = self._b_max * (1.0 - normalised ** self._gamma)
        return int(np.clip(b, self._b_min, self._b_max))

    def zone_brightnesses(self, zone_risks: list[float]) -> list[int]:
        """Convert all zone risk scores to brightness values."""
        return [self.brightness(r) for r in zone_risks]

    # Alias for API convenience
    compute_brightness = zone_brightnesses


if __name__ == "__main__":
    zm = ZoneMapper()
    print(f"Zones: {zm.zone_count}")
    # Test brightness mapping
    for risk in [0, 20, 40, 60, 80, 100]:
        b = zm.brightness(risk)
        print(f"  Risk={risk:3d} → Brightness={b:3d}")
    # Test zone assignment
    for px in [0, 160, 320, 640, 960, 1280]:
        z = zm.object_to_zone(px)
        print(f"  Pixel x={px:4d} → Zone {z}")
