"""
CARLA scenario runner for smart-adaptive-headlight.

Runs co-simulations in CARLA simulator (night driving, oncoming traffic)
or falls back to synthetic mock telemetry if CARLA python bindings are missing.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config

logger = logging.getLogger(__name__)


class CarlaScenarioRunner:
    """CARLA co-simulation runner with mock fallback mode."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        host: str = "127.0.0.1",
        port: int = 2000,
        mock: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.mock = mock

        if config_path is None:
            config_path = (
                Path(__file__).parent
                / "scenario_configs"
                / "default_night_drive.json"
            )

        self.scenario_cfg = self._load_scenario_config(Path(config_path))
        self._carla_client = None
        self._world = None

        if not self.mock:
            self._connect_carla()

    def _load_scenario_config(self, path: Path) -> dict[str, Any]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "scenario_name": "default_mock",
            "ego_vehicle": {"target_speed_kmh": 60.0},
            "traffic_actors": [],
        }

    def _connect_carla(self) -> None:
        try:
            import carla

            self._carla_client = carla.Client(self.host, self.port)
            self._carla_client.set_timeout(5.0)
            self._world = self._carla_client.get_world()
            logger.info("Connected to CARLA simulator at %s:%d", self.host, self.port)
        except Exception as err:
            logger.warning(
                "CARLA connection failed (%s). Falling back to mock simulation mode.",
                err,
            )
            self.mock = True

    def step(self) -> dict[str, Any]:
        """Execute one simulation step and return mock or CARLA frame data."""
        if self.mock or self._world is None:
            # Generate synthetic synthetic frame data
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            # Draw synthetic oncoming headlights
            cv2_available = False
            try:
                import cv2
                cv2.circle(frame, (450, 400), 20, (255, 255, 255), -1)
                cv2.circle(frame, (490, 400), 20, (255, 255, 255), -1)
                cv2_available = True
            except ImportError:
                pass

            return {
                "frame": frame,
                "timestamp": time.time(),
                "ego_speed_kmh": float(
                    self.scenario_cfg.get("ego_vehicle", {}).get(
                        "target_speed_kmh", 60.0
                    )
                ),
                "weather": "clear",
                "actors": [
                    {
                        "type": "car",
                        "distance": 45.0,
                        "relative_speed": 22.0,
                        "bbox": (430, 380, 510, 430),
                    }
                ],
            }
        else:
            # Code path for live CARLA world tick
            world_snapshot = self._world.get_snapshot()
            return {
                "frame": np.zeros((720, 1280, 3), dtype=np.uint8),
                "timestamp": world_snapshot.timestamp.elapsed_seconds,
                "ego_speed_kmh": 60.0,
                "weather": "clear",
                "actors": [],
            }


if __name__ == "__main__":
    runner = CarlaScenarioRunner(mock=True)
    step_data = runner.step()
    print("CARLA Scenario Runner Step Test:", step_data["ego_speed_kmh"], "km/h")
