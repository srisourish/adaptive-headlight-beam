"""
Serial bridge for smart-adaptive-headlight.

Implements the binary packet protocol:
  [SYNC_BYTE][ZONE_COUNT][ZONE_BRIGHTNESS...][PAN][TILT][CHECKSUM]

Handles serial reconnect logic and mock mode for testing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config

try:
    from diagnostics.health_monitor import get_monitor as _get_monitor
    _HAS_DIAGNOSTICS = True
except ImportError:
    _HAS_DIAGNOSTICS = False

SYNC_BYTE = 0xAA


class SerialBridge:
    """Sends zone brightness + servo commands to Arduino via serial."""

    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 115200,
        mock: bool = False,
    ) -> None:
        self._port = port
        self._baud = baudrate
        self._mock = mock
        self._serial = None
        self._connected = False
        self._last_send = 0.0
        self._error_count: int = 0   # cumulative errors for this session
        self._reconnect_count: int = 0
        self._watchdog_triggers: int = 0

        zones_cfg = get_config("zones")
        self._zone_count = zones_cfg.get("zone_count", 8)

        if not mock:
            self._connect()

    def _connect(self) -> bool:
        """Attempt to open the serial port."""
        try:
            import serial
            self._serial = serial.Serial(
                self._port, self._baud, timeout=1.0
            )
            time.sleep(2.0)  # Arduino reset delay
            self._connected = True
            print(f"[SerialBridge] Connected to {self._port}")
            return True
        except Exception as exc:
            print(f"[SerialBridge] Connection failed ({exc}); mock mode.")
            self._mock = True
            self._connected = False
            return False

    def send(
        self,
        zone_brightnesses: list[int],
        pan: int = 90,
        tilt: int = 90,
        watchdog_trigger: bool = False,
    ) -> bool:
        """Send a command packet to the Arduino.

        Packet format:
          Byte 0: SYNC_BYTE (0xAA)
          Byte 1: ZONE_COUNT
          Bytes 2..2+N-1: Zone brightness values (0-255)
          Byte 2+N: Pan servo angle (0-180)
          Byte 2+N+1: Tilt servo angle (0-180)
          Byte 2+N+2: XOR checksum of bytes 1..2+N+1

        Args:
            watchdog_trigger: Set True to simulate a watchdog fail-safe trigger
                              (used in testing degraded scenarios).

        Returns:
            True if sent successfully.
        """
        n = self._zone_count
        # Clamp values
        brightnesses = [max(0, min(255, b)) for b in zone_brightnesses[:n]]
        while len(brightnesses) < n:
            brightnesses.append(0)
        pan = max(0, min(180, pan))
        tilt = max(0, min(180, tilt))

        # Build packet
        payload = [n] + brightnesses + [pan, tilt]
        checksum = 0
        for b in payload:
            checksum ^= b
        packet = bytes([SYNC_BYTE] + payload + [checksum])

        # --- Health diagnostics instrumentation ---
        new_errors = 0
        if watchdog_trigger:
            self._watchdog_triggers += 1
            new_errors += 1

        if self._mock:
            if _HAS_DIAGNOSTICS:
                _get_monitor().record("serial_bridge", {"error_count": new_errors})
            self._last_send = time.time()
            return True

        if not self._connected:
            self._reconnect_count += 1
            new_errors += 1
            if not self._connect():
                if _HAS_DIAGNOSTICS:
                    _get_monitor().record("serial_bridge", {"error_count": new_errors})
                return False

        try:
            self._serial.write(packet)
            self._last_send = time.time()
            if _HAS_DIAGNOSTICS:
                _get_monitor().record("serial_bridge", {"error_count": new_errors})
            return True
        except Exception as exc:
            print(f"[SerialBridge] Write error ({exc}); reconnecting...")
            self._connected = False
            self._error_count += 1
            new_errors += 1
            if _HAS_DIAGNOSTICS:
                _get_monitor().record("serial_bridge", {"error_count": new_errors})
            return False

    def read_response(self) -> bytes | None:
        """Read any response from Arduino (if available)."""
        if self._mock or not self._connected:
            return None
        try:
            if self._serial.in_waiting > 0:
                return self._serial.read(self._serial.in_waiting)
        except Exception:
            pass
        return None

    def close(self) -> None:
        if self._serial and self._connected:
            self._serial.close()
            self._connected = False

    # Aliases & Status helper
    send_pwm = send

    def get_status(self) -> dict:
        return {
            "connected": self._connected or self._mock,
            "mock": self._mock,
            "port": self._port,
            "baudrate": self._baud,
            "packets_sent": 1 if self._last_send > 0 else 0,
        }

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_mock(self) -> bool:
        return self._mock

    @staticmethod
    def build_packet(zone_brightnesses: list[int], pan: int, tilt: int) -> bytes:
        """Build a packet without sending (useful for testing)."""
        n = len(zone_brightnesses)
        payload = [n] + list(zone_brightnesses) + [pan, tilt]
        checksum = 0
        for b in payload:
            checksum ^= b
        return bytes([SYNC_BYTE] + payload + [checksum])

    @staticmethod
    def verify_checksum(packet: bytes) -> bool:
        """Verify packet checksum."""
        if len(packet) < 4 or packet[0] != SYNC_BYTE:
            return False
        payload = packet[1:-1]
        checksum = 0
        for b in payload:
            checksum ^= b
        return checksum == packet[-1]


if __name__ == "__main__":
    bridge = SerialBridge(mock=True)
    brightnesses = [255, 200, 100, 50, 20, 50, 100, 200]
    ok = bridge.send(brightnesses, pan=90, tilt=85)
    print(f"[SerialBridge] Sent (mock): ok={ok}")
    pkt = SerialBridge.build_packet(brightnesses, 90, 85)
    print(f"  Packet ({len(pkt)} bytes): {pkt.hex()}")
    print(f"  Checksum valid: {SerialBridge.verify_checksum(pkt)}")
    bridge.close()
