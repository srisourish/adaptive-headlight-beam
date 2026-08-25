"""Unit test for Glare Risk Model (fusion.glare_risk_model)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fusion.glare_risk_model import (
    GlareFeatureVector,
    GlareRiskModel,
    HeuristicGlareRiskModel,
)


def test_heuristic_glare_risk_high_oncoming() -> None:
    """Test high glare risk score for close oncoming vehicle."""
    model = HeuristicGlareRiskModel()
    fv = GlareFeatureVector(
        object_type=2,  # car
        distance=15.0,  # 15 metres (very close)
        relative_speed=30.0,
        lane_position="oncoming",
        vertical_angle=0.0,
        vehicle_height_class=1,
        road_curvature=500.0,
        road_slope=0.0,
        weather_class=0,  # clear
        ambient_light=0.05,
        time_of_day=22.0,
    )

    risk = model.predict(fv)
    assert 50.0 <= risk <= 100.0, f"Expected high risk for close oncoming car, got {risk}"


def test_heuristic_glare_risk_distant() -> None:
    """Test low glare risk score for distant car."""
    model = HeuristicGlareRiskModel()
    fv = GlareFeatureVector(
        object_type=2,
        distance=180.0,  # 180 metres (beyond safe threshold)
        relative_speed=0.0,
        lane_position="adjacent_lane",
        vertical_angle=0.0,
        vehicle_height_class=1,
        road_curvature=1000.0,
        road_slope=0.0,
        weather_class=0,
        ambient_light=0.1,
        time_of_day=23.0,
    )

    risk = model.predict(fv)
    assert 0.0 <= risk < 30.0, f"Expected low risk for distant car, got {risk}"


def test_glare_risk_wrapper() -> None:
    """Test unified GlareRiskModel wrapper."""
    wrapper = GlareRiskModel()
    fv = GlareFeatureVector(
        object_type=2,
        distance=30.0,
        relative_speed=15.0,
        lane_position="same_lane_ahead",
        vertical_angle=0.0,
        vehicle_height_class=1,
        road_curvature=600.0,
        road_slope=0.0,
        weather_class=0,
        ambient_light=0.0,
        time_of_day=21.0,
    )
    score = wrapper.predict(fv)
    assert 0.0 <= score <= 100.0
