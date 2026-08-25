"""
Accident risk model for smart-adaptive-headlight.

Weighted linear combination of risk factors:
  speed, visibility, curvature, weather, hazard count.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config


class AccidentRiskModel:
    """Computes an aggregate accident/hazard risk score [0, 100]."""

    def __init__(self) -> None:
        cfg = get_config("thresholds")
        self._weights = cfg.get("accident_risk_weights", {})

    def predict(
        self,
        speed_kmh: float,
        visibility_m: float,
        curvature_radius: float,
        weather_class: int,
        hazard_count: int,
    ) -> float:
        """Compute accident risk score.

        Args:
            speed_kmh: Ego vehicle speed in km/h.
            visibility_m: Estimated visibility distance in metres.
            curvature_radius: Road curvature radius in metres.
            weather_class: WeatherClass int (0=clear..4=snow).
            hazard_count: Number of detected hazards/objects.

        Returns:
            Risk score [0, 100].
        """
        w = self._weights
        # Speed factor: higher speed = higher risk
        speed_score = np.clip(speed_kmh / 130.0, 0, 1)
        # Visibility: lower visibility = higher risk
        vis_score = 1.0 - np.clip(visibility_m / 300.0, 0, 1)
        # Curvature: tighter curves = higher risk
        curv_score = np.clip(1.0 - curvature_radius / 500.0, 0, 1)
        # Weather: higher class index = worse
        weather_score = np.clip(weather_class / 4.0, 0, 1)
        # Hazard count: more objects = more risk
        hazard_score = np.clip(hazard_count / 10.0, 0, 1)

        risk = (
            w.get("speed_factor", 0.25) * speed_score
            + w.get("visibility_factor", 0.25) * vis_score
            + w.get("curvature_factor", 0.20) * curv_score
            + w.get("weather_factor", 0.15) * weather_score
            + w.get("hazard_count_factor", 0.15) * hazard_score
        ) * 100.0

        return float(np.clip(risk, 0, 100))


if __name__ == "__main__":
    model = AccidentRiskModel()
    scenarios = [
        (60, 200, 500, 0, 2, "Normal driving"),
        (120, 50, 100, 2, 5, "Fast + fog + curve"),
        (30, 300, 9999, 0, 0, "Slow + clear + straight"),
    ]
    for spd, vis, curv, wea, haz, desc in scenarios:
        risk = model.predict(spd, vis, curv, wea, haz)
        print(f"  {desc:30s} → risk={risk:.1f}")
