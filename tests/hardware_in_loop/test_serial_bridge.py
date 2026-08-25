"""Hardware-in-the-Loop unit test for Serial Bridge (actuation.serial_bridge)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from actuation.serial_bridge import SerialBridge


def test_serial_bridge_mock_mode() -> None:
    """Test SerialBridge operates smoothly in mock mode without physical serial device."""
    bridge = SerialBridge(port="MOCK", baudrate=115200, mock=True)
    assert bridge.is_connected

    # Test packet transmission
    pwm_values = [255, 200, 150, 100, 50, 20, 10, 255]
    success = bridge.send_pwm(pwm_values)
    assert success is True

    # Test status check
    status = bridge.get_status()
    assert isinstance(status, dict)
    assert status["connected"] is True
    assert status["packets_sent"] >= 1

    bridge.close()
