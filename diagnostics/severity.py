"""
diagnostics/severity.py — Severity classification for system-health diagnostics.

Defines the three severity levels (INFO, WARNING, CRITICAL) and the HealthFinding
dataclass that every diagnostic check returns when it detects an anomaly.

NOTE: This module is intentionally separate from fusion/explainability.py.
      That module explains *driving decisions* (why a zone was dimmed).
      This module explains *pipeline health* (why the system might fail).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class HealthSeverity(Enum):
    """Diagnostic severity levels, ordered least to most severe."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        """Numeric rank for comparison (INFO=0, WARNING=1, CRITICAL=2)."""
        return {"INFO": 0, "WARNING": 1, "CRITICAL": 2}[self.value]

    def __gt__(self, other: "HealthSeverity") -> bool:
        return self.rank > other.rank

    def __ge__(self, other: "HealthSeverity") -> bool:
        return self.rank >= other.rank

    def __lt__(self, other: "HealthSeverity") -> bool:
        return self.rank < other.rank

    def __le__(self, other: "HealthSeverity") -> bool:
        return self.rank <= other.rank


@dataclass
class HealthFinding:
    """A single diagnostic finding produced by one of the 9 diagnostic checks.

    Fields
    ------
    signal : str
        The internal signal / check name (e.g. 'detection_confidence_drift').
    severity : HealthSeverity
        Severity of the finding.
    message : str
        Human-readable description of what was observed and why it matters.
    first_observed_frame : int
        Frame index at which this condition was first triggered.
    last_observed_frame : int
        Frame index at which this condition was most recently confirmed.
    recommended_action : str
        Short, concrete, non-alarmist suggestion for the operator/developer.
    metric_name : str
        The specific metric that triggered this finding (for sparkline labelling).
    metric_value : float
        The current value of that metric (raw, for display purposes).
    is_resolved : bool
        True once the condition falls back below threshold.
    """

    signal: str
    severity: HealthSeverity
    message: str
    first_observed_frame: int
    last_observed_frame: int
    recommended_action: str
    metric_name: str = ""
    metric_value: float = 0.0
    is_resolved: bool = False

    @property
    def duration_frames(self) -> int:
        """Number of frames this finding has been active."""
        return max(0, self.last_observed_frame - self.first_observed_frame)

    def to_dict(self) -> dict:
        """Return JSON-serialisable dict."""
        return {
            "signal": self.signal,
            "severity": self.severity.value,
            "message": self.message,
            "first_observed_frame": self.first_observed_frame,
            "last_observed_frame": self.last_observed_frame,
            "duration_frames": self.duration_frames,
            "recommended_action": self.recommended_action,
            "metric_name": self.metric_name,
            "metric_value": round(self.metric_value, 4),
            "is_resolved": self.is_resolved,
        }


def worst_severity(findings: list[HealthFinding]) -> Optional[HealthSeverity]:
    """Return the worst (highest-rank) severity from a list of findings."""
    if not findings:
        return None
    return max(f.severity for f in findings if not f.is_resolved)
