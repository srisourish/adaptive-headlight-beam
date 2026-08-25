"""
Beam state machine for smart-adaptive-headlight.

States: HIGH_BEAM, MEDIUM_BEAM, LOW_BEAM, MATRIX_PARTIAL
Transition rules with asymmetric hysteresis (fast-to-protect, slow-to-restore).
"""

from __future__ import annotations

import sys
import time
from enum import Enum
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config

try:
    from diagnostics.health_monitor import get_monitor as _get_monitor
    _HAS_DIAGNOSTICS = True
except ImportError:
    _HAS_DIAGNOSTICS = False


class BeamMode(Enum):
    HIGH_BEAM = "HIGH_BEAM"
    MEDIUM_BEAM = "MEDIUM_BEAM"
    LOW_BEAM = "LOW_BEAM"
    MATRIX_PARTIAL = "MATRIX_PARTIAL"


class BeamStateMachine:
    """Controls beam mode transitions based on per-zone risk scores."""

    def __init__(self) -> None:
        cfg = get_config("thresholds")
        beam_cfg = cfg.get("beam", {})
        self._min_speed = beam_cfg.get("min_speed_high_beam", 40)
        self._risk_low = beam_cfg.get("risk_threshold_low", 20)
        self._risk_high = beam_cfg.get("risk_threshold_high", 60)
        self._majority_frac = beam_cfg.get("majority_high_risk_fraction", 0.6)
        self._t_protect = beam_cfg.get("transition_to_protect_s", 0.15)
        self._t_restore = beam_cfg.get("transition_to_restore_s", 1.5)
        weather_pen = cfg.get("weather_penalties", {})
        self._bad_weather = {k for k, v in weather_pen.items() if v > 1.2}

        self._state = BeamMode.LOW_BEAM
        self._last_transition = time.time()
        self._pending_state: BeamMode | None = None
        self._pending_since: float = 0.0

    @property
    def state(self) -> BeamMode:
        return self._state

    def update(
        self,
        zone_risks: list[float],
        ego_speed_kmh: float = 50.0,
        weather: str = "clear",
    ) -> BeamMode:
        """Compute next beam state from zone risk array.

        Args:
            zone_risks: Per-zone risk scores [0, 100].
            ego_speed_kmh: Current vehicle speed.
            weather: Weather class name string.

        Returns:
            Current beam mode after transition logic.
        """
        now = time.time()
        prev_state = self._state
        target = self._compute_target(zone_risks, ego_speed_kmh, weather)

        if target == self._state:
            self._pending_state = None
            transitioned = False
        else:
            # Determine hysteresis delay
            is_protecting = self._is_more_protective(target)
            delay = self._t_protect if is_protecting else self._t_restore

            if self._pending_state != target:
                self._pending_state = target
                self._pending_since = now

            transitioned = False
            if now - self._pending_since >= delay:
                self._state = target
                self._last_transition = now
                self._pending_state = None
                transitioned = (prev_state != self._state)

        # --- Health diagnostics instrumentation ---
        if _HAS_DIAGNOSTICS:
            _get_monitor().record("beam_sm", {"transitioned": transitioned})

        return self._state

    def _compute_target(
        self, zone_risks: list[float], speed: float, weather: str
    ) -> BeamMode:
        """Determine the target beam mode (before hysteresis)."""
        if not zone_risks:
            return BeamMode.LOW_BEAM

        n = len(zone_risks)
        n_high = sum(1 for r in zone_risks if r > self._risk_high)
        all_low = all(r < self._risk_low for r in zone_risks)

        # Bad weather → force LOW_BEAM
        if weather in self._bad_weather:
            return BeamMode.LOW_BEAM

        # Majority high-risk → LOW_BEAM
        if n_high / n >= self._majority_frac:
            return BeamMode.LOW_BEAM

        # All zones safe + sufficient speed → HIGH_BEAM
        if all_low and speed >= self._min_speed:
            return BeamMode.HIGH_BEAM

        # All zones safe but slow → MEDIUM_BEAM
        if all_low and speed < self._min_speed:
            return BeamMode.MEDIUM_BEAM

        # Some zones have risk → MATRIX_PARTIAL
        return BeamMode.MATRIX_PARTIAL

    def _is_more_protective(self, target: BeamMode) -> bool:
        """True if transitioning to a more protective (dimmer) mode."""
        order = {
            BeamMode.HIGH_BEAM: 0,
            BeamMode.MEDIUM_BEAM: 1,
            BeamMode.MATRIX_PARTIAL: 2,
            BeamMode.LOW_BEAM: 3,
        }
        return order.get(target, 0) > order.get(self._state, 0)

    def force_state(self, mode: BeamMode) -> None:
        """Force a specific beam state (for override handler)."""
        self._state = mode
        self._pending_state = None
        self._last_transition = time.time()


if __name__ == "__main__":
    bsm = BeamStateMachine()
    scenarios = [
        ([0, 0, 0, 0, 0, 0, 0, 0], 60, "clear", "All safe + fast"),
        ([0, 0, 80, 90, 0, 0, 0, 0], 60, "clear", "Two zones high risk"),
        ([70, 80, 90, 85, 75, 80, 70, 60], 60, "clear", "Majority high risk"),
        ([0, 0, 0, 0, 0, 0, 0, 0], 20, "clear", "All safe + slow"),
        ([0, 0, 0, 0, 0, 0, 0, 0], 60, "fog", "Safe but fog"),
    ]
    for risks, speed, weather, desc in scenarios:
        # Reset state
        bsm._state = BeamMode.LOW_BEAM
        bsm._pending_state = None
        # Simulate rapid updates to bypass hysteresis
        for _ in range(20):
            mode = bsm.update(risks, speed, weather)
            time.sleep(0.1)
        print(f"  {desc:35s} → {mode.value}")
