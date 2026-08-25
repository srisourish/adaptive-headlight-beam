"""
Override handler for smart-adaptive-headlight.

Handles manual override input:
  - Keyboard key in demo mode
  - GPIO pin on real hardware
Force-Low is always allowed; force-High capped by a safety ceiling.
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config


class OverrideMode(Enum):
    NONE = "none"
    FORCE_LOW = "force_low"
    FORCE_HIGH = "force_high"


class OverrideHandler:
    """Manages manual beam override state."""

    def __init__(self) -> None:
        cfg = get_config("thresholds")
        override_cfg = cfg.get("override", {})
        self._ceiling = override_cfg.get("force_high_ceiling", 180)
        self._toggle_key = override_cfg.get("toggle_key", "h")
        self._mode = OverrideMode.NONE
        self._key_states = {"l": False, "h": False}

    @property
    def mode(self) -> OverrideMode:
        return self._mode

    @property
    def force_high_ceiling(self) -> int:
        return self._ceiling

    def handle_key(self, key: str) -> OverrideMode:
        """Process a keyboard input for override control.

        Args:
            key: Single character key press.

        Returns:
            Current override mode after processing.
        """
        if key == "l":
            # Toggle force-low
            if self._mode == OverrideMode.FORCE_LOW:
                self._mode = OverrideMode.NONE
            else:
                self._mode = OverrideMode.FORCE_LOW
        elif key == self._toggle_key:
            # Toggle force-high (with safety ceiling)
            if self._mode == OverrideMode.FORCE_HIGH:
                self._mode = OverrideMode.NONE
            else:
                self._mode = OverrideMode.FORCE_HIGH
        return self._mode

    def apply_override(
        self, zone_brightnesses: list[int], has_objects: bool
    ) -> list[int]:
        """Apply override to zone brightness values.

        Args:
            zone_brightnesses: Computed brightness per zone.
            has_objects: Whether any objects are detected.

        Returns:
            Modified brightness values.
        """
        if self._mode == OverrideMode.FORCE_LOW:
            # Force all zones to minimum
            return [10] * len(zone_brightnesses)

        if self._mode == OverrideMode.FORCE_HIGH:
            if has_objects:
                # Cap at safety ceiling
                return [min(b, self._ceiling) for b in zone_brightnesses]
            else:
                # No objects — allow full brightness
                return [255] * len(zone_brightnesses)

        return zone_brightnesses

    def reset(self) -> None:
        self._mode = OverrideMode.NONE


if __name__ == "__main__":
    oh = OverrideHandler()
    print(f"Initial mode: {oh.mode.value}")
    oh.handle_key("l")
    print(f"After 'l': {oh.mode.value}")
    result = oh.apply_override([200, 180, 100, 50, 30, 100, 200, 255], True)
    print(f"  Force-low brightness: {result}")
    oh.handle_key("l")  # Toggle off
    oh.handle_key("h")
    print(f"After 'h': {oh.mode.value}")
    result = oh.apply_override([200, 180, 100, 50, 30, 100, 200, 255], True)
    print(f"  Force-high (with objects): {result}")
    result = oh.apply_override([200, 180, 100, 50, 30, 100, 200, 255], False)
    print(f"  Force-high (no objects): {result}")
