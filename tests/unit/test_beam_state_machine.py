"""Unit test for Beam State Machine (decision.beam_state_machine)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from decision.beam_state_machine import BeamMode, BeamStateMachine


def test_state_machine_low_speed() -> None:
    """At low speed (<40 km/h), beam state should remain LOW_BEAM regardless of zero risk."""
    sm = BeamStateMachine()
    mode = sm.update(zone_risks=[0.0] * 8, ego_speed_kmh=20.0)
    assert mode == BeamMode.LOW_BEAM


def test_state_machine_high_beam_trigger() -> None:
    """High speed with zero risk should transition to HIGH_BEAM after restore delay."""
    sm = BeamStateMachine()
    # Initial state is LOW_BEAM
    assert sm.state == BeamMode.LOW_BEAM

    # Update with zero risk and high speed
    sm.update(zone_risks=[0.0] * 8, ego_speed_kmh=65.0)

    # Fast forward time to bypass restore hysteresis delay
    sm._last_transition = time.time() - 2.0
    sm._pending_since = time.time() - 2.0

    mode = sm.update(zone_risks=[0.0] * 8, ego_speed_kmh=65.0)
    assert mode in (BeamMode.HIGH_BEAM, BeamMode.MATRIX_PARTIAL)


def test_state_machine_protect_hysteresis() -> None:
    """High risk triggers fast dimming protection."""
    sm = BeamStateMachine()
    # Force state to HIGH_BEAM
    sm._state = BeamMode.HIGH_BEAM

    # Present high risk across all zones
    mode = sm.update(zone_risks=[85.0] * 8, ego_speed_kmh=65.0)

    # Bypass fast protect delay (0.15s)
    sm._pending_since = time.time() - 0.5
    mode = sm.update(zone_risks=[85.0] * 8, ego_speed_kmh=65.0)
    assert mode == BeamMode.LOW_BEAM
