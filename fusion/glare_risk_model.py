"""
Glare Risk Model for smart-adaptive-headlight.

Dual-backend architecture:
  1. Hand-tuned heuristic formula (weighted sub-scores) — default/fallback.
  2. GBMGlareRiskModel wrapping a trained XGBoost/LightGBM model.

Both share the same ``build_feature_vector()`` and ``predict()`` interface.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config


@dataclass
class GlareFeatureVector:
    """Exact feature vector for glare risk prediction."""
    object_type: int           # 0=person, 2=car, 7=truck, etc.
    distance: float            # metres
    relative_speed: float      # m/s (positive = closing)
    lane_position: str         # LaneRole value string
    vertical_angle: float      # degrees (object above/below horizon)
    vehicle_height_class: int  # 0=low, 1=medium, 2=high (sedan/SUV/truck)
    road_curvature: float      # curvature radius in metres
    road_slope: float          # degrees
    weather_class: int         # WeatherClass int
    ambient_light: float       # 0=dark, 1=bright
    time_of_day: float         # 0-24 hours

    def to_array(self) -> np.ndarray:
        """Convert to numeric array for model input."""
        lane_map = {
            "oncoming": 1.0, "same_lane_ahead": 0.7,
            "adjacent_lane": 0.4, "parked": 0.1, "irrelevant": 0.0,
        }
        return np.array([
            float(self.object_type),
            self.distance,
            self.relative_speed,
            lane_map.get(self.lane_position, 0.0),
            self.vertical_angle,
            float(self.vehicle_height_class),
            self.road_curvature,
            self.road_slope,
            float(self.weather_class),
            self.ambient_light,
            self.time_of_day,
        ], dtype=np.float32)


def build_feature_vector(
    object_type: int,
    distance: float,
    relative_speed: float,
    lane_position: str,
    vertical_angle: float,
    vehicle_height_class: int,
    road_curvature: float,
    road_slope: float,
    weather_class: int,
    ambient_light: float,
    time_of_day: float,
) -> GlareFeatureVector:
    """Build a typed feature vector from raw values."""
    return GlareFeatureVector(
        object_type=object_type, distance=distance,
        relative_speed=relative_speed, lane_position=lane_position,
        vertical_angle=vertical_angle,
        vehicle_height_class=vehicle_height_class,
        road_curvature=road_curvature, road_slope=road_slope,
        weather_class=weather_class, ambient_light=ambient_light,
        time_of_day=time_of_day,
    )


class HeuristicGlareRiskModel:
    """Hand-tuned glare risk scoring (0-100)."""

    def __init__(self) -> None:
        cfg = get_config("thresholds")
        self._weights = cfg.get("glare_risk_weights", {})
        self._weather_pen = cfg.get("weather_penalties", {})
        depth_cfg = cfg.get("depth", {})
        self._max_depth = depth_cfg.get("max_depth_m", 200.0)
        self._min_depth = depth_cfg.get("min_depth_m", 5.0)

    def predict(self, fv: GlareFeatureVector) -> float:
        """Compute glare risk score [0, 100]."""
        w = self._weights

        # Proximity: closer = higher risk (inverse distance, normalised)
        prox = 1.0 - np.clip(
            (fv.distance - self._min_depth) / (self._max_depth - self._min_depth),
            0, 1,
        )
        # Closing speed: positive = approaching
        speed_norm = np.clip(fv.relative_speed / 30.0, 0, 1)
        # Lane relevance
        lane_map = {
            "oncoming": 1.0, "same_lane_ahead": 0.6,
            "adjacent_lane": 0.3, "parked": 0.05, "irrelevant": 0.0,
        }
        lane_score = lane_map.get(fv.lane_position, 0.0)
        # Eye exposure (vertical angle — high headlights more blinding)
        eye_exp = np.clip(1.0 - abs(fv.vertical_angle) / 10.0, 0, 1)
        # Weather penalty
        weather_names = {0: "clear", 1: "rain", 2: "fog", 3: "dust", 4: "snow"}
        wname = weather_names.get(fv.weather_class, "clear")
        weather_mult = self._weather_pen.get(wname, 1.0)
        weather_score = (weather_mult - 1.0) / 0.6  # normalise [0,1]
        weather_score = np.clip(weather_score, 0, 1)
        # Curvature anticipation (tight curves increase risk)
        curv_score = np.clip(1.0 - fv.road_curvature / 500.0, 0, 1)
        # Vehicle height class
        height_score = fv.vehicle_height_class / 2.0
        # Vertical angle component
        vert_score = np.clip(abs(fv.vertical_angle) / 15.0, 0, 1)

        # Weighted sum
        raw = (
            w.get("proximity", 0.25) * prox
            + w.get("closing_speed", 0.15) * speed_norm
            + w.get("lane_relevance", 0.20) * lane_score
            + w.get("eye_exposure", 0.10) * eye_exp
            + w.get("weather_penalty", 0.10) * weather_score
            + w.get("curvature_anticipation", 0.10) * curv_score
            + w.get("vertical_angle", 0.05) * vert_score
            + w.get("vehicle_height_class", 0.05) * height_score
        )

        # Apply weather multiplier on top
        risk = raw * 100.0 * weather_mult
        return float(np.clip(risk, 0, 100))

    def feature_contributions(self, fv: GlareFeatureVector) -> dict[str, float]:
        """Return per-feature contribution for explainability."""
        w = self._weights
        prox = 1.0 - np.clip((fv.distance - self._min_depth) / (self._max_depth - self._min_depth), 0, 1)
        speed_norm = np.clip(fv.relative_speed / 30.0, 0, 1)
        lane_map = {"oncoming": 1.0, "same_lane_ahead": 0.6, "adjacent_lane": 0.3, "parked": 0.05, "irrelevant": 0.0}
        lane_score = lane_map.get(fv.lane_position, 0.0)
        return {
            "proximity": w.get("proximity", 0.25) * prox,
            "closing_speed": w.get("closing_speed", 0.15) * speed_norm,
            "lane_relevance": w.get("lane_relevance", 0.20) * lane_score,
            "distance_m": fv.distance,
            "lane_position": fv.lane_position,
        }


class GBMGlareRiskModel:
    """XGBoost/LightGBM-backed glare risk model."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model = None
        self._feature_names = [
            "object_type", "distance", "relative_speed", "lane_position",
            "vertical_angle", "vehicle_height_class", "road_curvature",
            "road_slope", "weather_class", "ambient_light", "time_of_day",
        ]
        if model_path:
            self._load(model_path)

    def _load(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            self._model = pickle.load(f)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, fv: GlareFeatureVector) -> float:
        if self._model is None:
            raise RuntimeError("GBM model not loaded")
        x = fv.to_array().reshape(1, -1)
        pred = float(self._model.predict(x)[0])
        return float(np.clip(pred, 0, 100))

    def feature_importances(self) -> dict[str, float]:
        if self._model is None:
            return {}
        try:
            imp = self._model.feature_importances_
            return dict(zip(self._feature_names, imp.tolist()))
        except AttributeError:
            return {}


class GlareRiskPredictor:
    """Unified predictor — uses GBM if available, else heuristic."""

    def __init__(self, gbm_path: Optional[str] = None) -> None:
        self._heuristic = HeuristicGlareRiskModel()
        self._gbm = GBMGlareRiskModel(gbm_path)

    def predict(self, fv: GlareFeatureVector) -> float:
        if self._gbm.is_loaded:
            return self._gbm.predict(fv)
        return self._heuristic.predict(fv)

    @property
    def backend(self) -> str:
        return "gbm" if self._gbm.is_loaded else "heuristic"

    def contributions(self, fv: GlareFeatureVector) -> dict[str, float]:
        if self._gbm.is_loaded:
            return self._gbm.feature_importances()
        return self._heuristic.feature_contributions(fv)


# Alias for backward compatibility and test consistency
GlareRiskModel = GlareRiskPredictor


if __name__ == "__main__":
    predictor = GlareRiskPredictor()
    print(f"Backend: {predictor.backend}")
    fv = build_feature_vector(
        object_type=2, distance=25.0, relative_speed=15.0,
        lane_position="oncoming", vertical_angle=-2.0,
        vehicle_height_class=1, road_curvature=300.0,
        road_slope=0.0, weather_class=0, ambient_light=0.1,
        time_of_day=22.0,
    )
    risk = predictor.predict(fv)
    print(f"Glare risk: {risk:.1f}/100")
    print(f"Contributions: {predictor.contributions(fv)}")
