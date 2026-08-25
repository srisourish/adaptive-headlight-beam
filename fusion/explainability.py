"""
Explainability module for smart-adaptive-headlight.

Logs top contributing features per decision into a structured format
consumable by the dashboard. Supports both heuristic (weighted terms)
and GBM (feature importance) backends.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("explainability")


@dataclass
class ExplainabilityEntry:
    """Single decision explanation."""
    timestamp: float
    zone_id: int
    risk_score: float
    beam_mode: str
    top_features: list[dict[str, float]]  # [{name: contribution}, ...]
    track_id: Optional[int] = None
    object_type: Optional[str] = None
    distance_m: Optional[float] = None
    lane_position: Optional[str] = None


class ExplainabilityLogger:
    """Accumulates and serves explanation entries for the dashboard."""

    def __init__(self, max_history: int = 200) -> None:
        self._history: list[ExplainabilityEntry] = []
        self._max = max_history

    def log_decision(
        self,
        zone_id: int,
        risk_score: float,
        beam_mode: str,
        contributions: dict[str, float],
        top_k: int = 5,
        track_id: Optional[int] = None,
        object_type: Optional[str] = None,
        distance_m: Optional[float] = None,
        lane_position: Optional[str] = None,
    ) -> ExplainabilityEntry:
        """Log a beam decision with feature contributions.

        Args:
            zone_id: Zone index.
            risk_score: Computed risk [0, 100].
            beam_mode: Current beam state name.
            contributions: {feature_name: contribution_value}.
            top_k: Number of top features to retain.
            track_id: Associated track ID (if per-object).
            object_type: Class name of the triggering object.
            distance_m: Distance to object in metres.
            lane_position: Lane role string.
        """
        # Sort by absolute contribution, take top-k
        sorted_feats = sorted(
            contributions.items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:top_k]
        top_features = [{"name": k, "value": v} for k, v in sorted_feats]

        entry = ExplainabilityEntry(
            timestamp=time.time(),
            zone_id=zone_id,
            risk_score=risk_score,
            beam_mode=beam_mode,
            top_features=top_features,
            track_id=track_id,
            object_type=object_type,
            distance_m=distance_m,
            lane_position=lane_position,
        )

        self._history.append(entry)
        if len(self._history) > self._max:
            self._history = self._history[-self._max:]

        logger.debug(
            "Zone %d risk=%.1f mode=%s top=%s",
            zone_id, risk_score, beam_mode,
            json.dumps(top_features),
        )
        return entry

    def get_latest(self, n: int = 1) -> list[dict]:
        """Return the latest n entries as dicts (JSON-serialisable)."""
        entries = self._history[-n:]
        return [asdict(e) for e in entries]

    def get_by_zone(self, zone_id: int) -> Optional[dict]:
        """Return the most recent entry for a given zone."""
        for entry in reversed(self._history):
            if entry.zone_id == zone_id:
                return asdict(entry)
        return None

    def clear(self) -> None:
        self._history.clear()


if __name__ == "__main__":
    el = ExplainabilityLogger()
    el.log_decision(
        zone_id=3, risk_score=72.5, beam_mode="MATRIX_PARTIAL",
        contributions={
            "proximity": 0.22, "lane_relevance": 0.18,
            "closing_speed": 0.12, "weather_penalty": 0.08,
            "eye_exposure": 0.05, "curvature": 0.03,
        },
        track_id=7, object_type="car", distance_m=22.0,
        lane_position="oncoming",
    )
    latest = el.get_latest(1)
    print(json.dumps(latest, indent=2))
