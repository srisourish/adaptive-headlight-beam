"""
Training script for Glare Risk Model.

Generates synthetic telemetry data or loads CSV dataset to train
a Gradient Boosting Regressor (or Random Forest fallback) for glare risk prediction.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fusion.glare_risk_model import GlareFeatureVector, HeuristicGlareRiskModel


def generate_synthetic_dataset(n_samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic telemetry samples labeled by heuristic model."""
    heuristic = HeuristicGlareRiskModel()
    X = []
    y = []

    np.random.seed(42)
    for _ in range(n_samples):
        obj_type = np.random.choice([0, 2, 7])
        dist = np.random.uniform(5.0, 200.0)
        rel_speed = np.random.uniform(-10.0, 40.0)
        lane_pos = np.random.choice(
            ["oncoming", "same_lane_ahead", "adjacent_lane", "parked"]
        )
        vert_angle = np.random.uniform(-5.0, 5.0)
        v_class = np.random.choice([0, 1, 2])
        curvature = np.random.uniform(50.0, 1000.0)
        slope = np.random.uniform(-5.0, 5.0)
        weather_cls = np.random.choice([0, 1, 2, 3])
        amb_light = np.random.uniform(0.0, 0.5)
        time_tod = np.random.uniform(20.0, 24.0)

        fv = GlareFeatureVector(
            object_type=obj_type,
            distance=dist,
            relative_speed=rel_speed,
            lane_position=lane_pos,
            vertical_angle=vert_angle,
            vehicle_height_class=v_class,
            road_curvature=curvature,
            road_slope=slope,
            weather_class=weather_cls,
            ambient_light=amb_light,
            time_of_day=time_tod,
        )

        score = heuristic.predict(fv)
        X.append(fv.to_array())
        y.append(score)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_model(
    output_path: str = "models/weights/glare_risk_gbm.pkl",
    n_samples: int = 1000,
) -> None:
    """Train gradient boosting or decision tree regressor model."""
    print(f"Generating {n_samples} synthetic training samples...")
    X, y = generate_synthetic_dataset(n_samples)

    try:
        from sklearn.ensemble import GradientBoostingRegressor

        regressor = GradientBoostingRegressor(n_estimators=50, max_depth=4)
        regressor.fit(X, y)
        train_score = regressor.score(X, y)
        print(f"GradientBoostingRegressor trained. R^2 score: {train_score:.4f}")
    except ImportError:
        print("scikit-learn not available. Saving a baseline linear model matrix.")
        # Minimal linear regression baseline using least squares: w = (X^T X)^-1 X^T y
        w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        regressor = {"weights": w, "bias": 0.0}

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(regressor, f)

    print(f"Saved glare risk model weights to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Glare Risk Model")
    parser.add_argument(
        "--output", default="models/weights/glare_risk_gbm.pkl", help="Output path"
    )
    parser.add_argument(
        "--samples", type=int, default=1000, help="Number of synthetic samples"
    )
    args = parser.parse_args()
    train_model(output_path=args.output, n_samples=args.samples)
