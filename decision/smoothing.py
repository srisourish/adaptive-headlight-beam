"""
Temporal smoothing / hysteresis filter for smart-adaptive-headlight.

Applies a hold-time + EMA filter to per-zone risk scores to prevent
rapid flickering of beam zones.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config


class RiskSmoother:
    """Temporal hold + EMA smoothing for per-zone risk values."""

    def __init__(self, zone_count: int | None = None) -> None:
        cfg = get_config("thresholds")
        smooth_cfg = cfg.get("smoothing", {})
        self._t_hold = smooth_cfg.get("t_hold", 1.0)
        self._alpha = smooth_cfg.get("ema_alpha", 0.3)

        zones_cfg = get_config("zones")
        n = zone_count or zones_cfg.get("zone_count", 8)
        self._n = n

        # Per-zone state
        self._smoothed = [0.0] * n
        self._held_until = [0.0] * n  # timestamp until which risk is held high
        self._peak = [0.0] * n

        beam_cfg = cfg.get("beam", {})
        self._risk_high = beam_cfg.get("risk_threshold_high", 60)

    def update(self, zone_risks: list[float]) -> list[float]:
        """Apply smoothing to raw per-zone risk scores.

        Args:
            zone_risks: Raw risk values per zone [0, 100].

        Returns:
            Smoothed risk values per zone [0, 100].
        """
        now = time.time()
        result = []

        for i in range(self._n):
            raw = zone_risks[i] if i < len(zone_risks) else 0.0

            # If risk exceeds high threshold, start/extend hold
            if raw > self._risk_high:
                self._held_until[i] = now + self._t_hold
                self._peak[i] = max(self._peak[i], raw)

            # During hold period, don't let risk drop below peak
            if now < self._held_until[i]:
                effective = max(raw, self._peak[i] * 0.8)
            else:
                effective = raw
                self._peak[i] = raw  # Reset peak

            # Exponential moving average
            self._smoothed[i] = (
                self._alpha * effective + (1 - self._alpha) * self._smoothed[i]
            )
            result.append(float(np.clip(self._smoothed[i], 0, 100)))

        return result

    def reset(self) -> None:
        """Reset all smoothing state."""
        self._smoothed = [0.0] * self._n
        self._held_until = [0.0] * self._n
        self._peak = [0.0] * self._n


if __name__ == "__main__":
    smoother = RiskSmoother(zone_count=4)
    print("[Smoothing] Simulating risk spike then drop:")
    # Simulate: risk spike at t=0, drop at t=0.5s
    for step in range(20):
        if step < 5:
            raw = [80.0, 0.0, 0.0, 0.0]
        else:
            raw = [10.0, 0.0, 0.0, 0.0]
        smoothed = smoother.update(raw)
        print(f"  Step {step:2d} raw={raw[0]:5.1f} → smoothed={smoothed[0]:5.1f}")
        time.sleep(0.15)
