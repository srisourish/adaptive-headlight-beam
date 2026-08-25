"""
diagnostics/health_monitor.py — Rolling-window signal collection and diagnostic checks.

This is the central engine of the System-Health Explainability module.

Architecture
------------
1. ``record(stage_name, metrics_dict)`` — lightweight hook called by each pipeline
   stage after it runs. Appends new metric values to per-signal rolling deques.
2. Nine diagnostic check functions, one per monitored signal. Each check inspects
   its rolling buffer and returns None (no issue) or a HealthFinding.
3. ``evaluate()`` — runs all checks, deduplicates findings, returns the active list.
4. ``get_metric_history(metric_name)`` — returns a list of recent values for
   sparkline visualisation in the dashboard.

Deduplication / cooldown
------------------------
A finding for signal S does NOT re-fire every frame once triggered. Instead, the
monitor tracks ``_active_findings`` keyed by signal name. If a check continues to
trigger, ``last_observed_frame`` is updated in-place. Once the condition clears, the
finding is marked ``is_resolved = True`` and removed from the active dict after a
configurable cooldown period.

NOTE: All numeric thresholds here are *illustrative defaults requiring tuning
against real operating data* before deployment.  They are exposed in
``config/thresholds.yaml`` under the ``health_diagnostics`` key so operators can
adjust them without touching source code.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Any, Deque, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config
from diagnostics.health_explainer import HealthExplainer
from diagnostics.severity import HealthFinding, HealthSeverity


# ---------------------------------------------------------------------------
# Default threshold constants (overridden by config/thresholds.yaml)
# These are ILLUSTRATIVE defaults — tune against real data before production.
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "window_size": 40,                # rolling window length in frames
    "cooldown_frames": 30,            # frames to keep a resolved finding visible
    "detection_confidence_min": 0.55, # rolling avg conf below this → warning
    "detection_dropout_max": 0.40,    # fraction of frames w/ 0 detections → warning
    "depth_clip_boundary_max": 0.35,  # fraction of depth samples at clip bounds → warning
    "tracker_churn_max": 0.60,        # new-track-ID spawns per detection → warning
    "latency_max_ms": 120.0,          # rolling avg frame latency (ms) → warning
    "serial_error_max": 1,            # cumulative serial errors → warning
    "weather_fallback_max": 0.50,     # fraction of frames using rule-based fallback → warning
    "beam_oscillation_max": 0.15,     # beam transitions per frame → warning
    "explainability_gap_max": 0.30,   # high-risk decisions w/ low attribution → warning
}


def _load_thresholds() -> dict[str, Any]:
    """Load health_diagnostics block from thresholds.yaml, fall back to defaults."""
    try:
        cfg = get_config("thresholds")
        hd = cfg.get("health_diagnostics", {})
        merged = dict(_DEFAULTS)
        merged.update({k: v for k, v in hd.items() if k in _DEFAULTS})
        return merged
    except Exception:
        return dict(_DEFAULTS)


class HealthMonitor:
    """
    Monitors rolling pipeline signals and fires diagnostic findings.

    Parameters
    ----------
    window_size : int | None
        Override the rolling window length. If None, uses config value.
    """

    def __init__(self, window_size: Optional[int] = None) -> None:
        self._cfg = _load_thresholds()
        self._win = window_size or int(self._cfg["window_size"])
        self._frame = 0  # monotone frame counter

        # Rolling signal buffers — each is a deque of recent scalar values
        self._buffers: dict[str, Deque[float]] = {
            "detection_conf":       deque(maxlen=self._win),
            "detection_zero":       deque(maxlen=self._win),  # 1.0 = zero detections
            "depth_clip":           deque(maxlen=self._win),  # fraction at clip bounds
            "track_churn":          deque(maxlen=self._win),  # new IDs / detections
            "frame_latency_ms":     deque(maxlen=self._win),
            "serial_errors":        deque(maxlen=self._win),  # cumulative counter sampled each frame
            "weather_fallback":     deque(maxlen=self._win),  # 1.0 = used fallback
            "beam_transitions":     deque(maxlen=self._win),  # 1.0 = transition occurred
            "attribution_gap":      deque(maxlen=self._win),  # 1.0 = attribution gap detected
        }

        # State tracking
        self._serial_error_total: int = 0
        self._active_findings: dict[str, HealthFinding] = {}  # signal → finding
        self._resolved_findings: dict[str, HealthFinding] = {}  # recently resolved
        self._resolved_at: dict[str, int] = {}  # signal → frame when resolved

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, stage: str, metrics: dict[str, Any]) -> None:
        """
        Record metrics from a pipeline stage.

        Called by each pipeline module after it runs. Unknown keys are silently ignored
        so adding new metrics never breaks existing callers.

        Standard metric keys per stage
        --------------------------------
        detector      : conf (float), n_detections (int)
        depth         : clip_fraction (float)
        tracker       : n_new_ids (int), n_detections (int)
        weather       : used_fallback (bool)
        serial_bridge : error_count (int)
        beam_sm       : transitioned (bool)
        fusion        : attribution_gap (bool)
        pipeline      : latency_ms (float)
        """
        m = metrics
        if stage == "detector":
            conf = float(m.get("conf", 0.0))
            n = int(m.get("n_detections", 0))
            if n > 0 or conf > 0:
                self._buffers["detection_conf"].append(max(conf, 0.0))
            self._buffers["detection_zero"].append(0.0 if n > 0 else 1.0)

        elif stage == "depth":
            self._buffers["depth_clip"].append(float(m.get("clip_fraction", 0.0)))

        elif stage == "tracker":
            n_new = int(m.get("n_new_ids", 0))
            n_det = max(int(m.get("n_detections", 0)), 1)  # avoid /0
            self._buffers["track_churn"].append(n_new / n_det)

        elif stage == "weather":
            self._buffers["weather_fallback"].append(1.0 if m.get("used_fallback", False) else 0.0)

        elif stage == "serial_bridge":
            delta = int(m.get("error_count", 0))
            self._serial_error_total += delta
            self._buffers["serial_errors"].append(float(self._serial_error_total))

        elif stage == "beam_sm":
            self._buffers["beam_transitions"].append(1.0 if m.get("transitioned", False) else 0.0)

        elif stage == "fusion":
            self._buffers["attribution_gap"].append(1.0 if m.get("attribution_gap", False) else 0.0)

        elif stage == "pipeline":
            self._buffers["frame_latency_ms"].append(float(m.get("latency_ms", 0.0)))

    def tick(self) -> None:
        """Advance the internal frame counter. Call once per processed frame."""
        self._frame += 1

    def evaluate(self) -> list[HealthFinding]:
        """
        Run all 9 diagnostic checks and return the list of active (non-resolved) findings.

        Findings are deduplicated — a persistent condition updates its
        ``last_observed_frame`` rather than creating a new entry.
        """
        checks = [
            self._check_confidence_drift,
            self._check_detection_dropout,
            self._check_depth_anomaly,
            self._check_tracker_churn,
            self._check_latency_trend,
            self._check_serial_health,
            self._check_weather_fallback,
            self._check_beam_oscillation,
            self._check_attribution_gap,
        ]

        newly_resolved: list[str] = []

        for check in checks:
            finding = check()
            if finding is None:
                # Check passed — mark any active finding for this signal as resolved
                signal = _check_to_signal(check.__name__)
                if signal in self._active_findings:
                    self._active_findings[signal].is_resolved = True
                    self._resolved_findings[signal] = self._active_findings[signal]
                    self._resolved_at[signal] = self._frame
                    newly_resolved.append(signal)
            else:
                signal = finding.signal
                if signal in self._active_findings:
                    # Update existing — do not re-fire
                    self._active_findings[signal].last_observed_frame = self._frame
                    self._active_findings[signal].metric_value = finding.metric_value
                    self._active_findings[signal].is_resolved = False
                else:
                    self._active_findings[signal] = finding

        # Remove resolved findings from active after cooldown
        for signal in list(newly_resolved):
            if signal in self._active_findings:
                del self._active_findings[signal]

        # Also expire old resolved findings beyond cooldown
        cooldown = int(self._cfg["cooldown_frames"])
        for sig in list(self._resolved_at.keys()):
            if self._frame - self._resolved_at[sig] > cooldown:
                self._resolved_findings.pop(sig, None)
                del self._resolved_at[sig]

        return list(self._active_findings.values())

    def get_metric_history(self, metric_name: str) -> list[float]:
        """Return recent values for a named buffer (for sparkline charts)."""
        return list(self._buffers.get(metric_name, []))

    def get_all_histories(self) -> dict[str, list[float]]:
        """Return all rolling buffers as plain lists."""
        return {k: list(v) for k, v in self._buffers.items()}

    @property
    def frame(self) -> int:
        return self._frame

    @property
    def active_findings(self) -> dict[str, HealthFinding]:
        return dict(self._active_findings)

    # ------------------------------------------------------------------
    # Internal diagnostic checks — one per monitored signal
    # ------------------------------------------------------------------

    def _check_confidence_drift(self) -> Optional[HealthFinding]:
        """Check 1: Rolling average detection confidence dropping below threshold."""
        buf = self._buffers["detection_conf"]
        if len(buf) < self._win // 2:
            return None  # not enough data yet
        avg = sum(buf) / len(buf)
        threshold = float(self._cfg["detection_confidence_min"])
        if avg < threshold:
            first = self._active_findings.get(
                "detection_confidence_drift",
                HealthFinding(signal="", severity=HealthSeverity.INFO,
                              message="", first_observed_frame=self._frame,
                              last_observed_frame=self._frame, recommended_action=""),
            ).first_observed_frame
            return HealthExplainer.explain_confidence_drift(avg, threshold, self._frame, first)
        return None

    def _check_detection_dropout(self) -> Optional[HealthFinding]:
        """Check 2: Fraction of recent frames with zero detections exceeds threshold."""
        buf = self._buffers["detection_zero"]
        if len(buf) < self._win // 2:
            return None
        rate = sum(buf) / len(buf)
        threshold = float(self._cfg["detection_dropout_max"])
        if rate > threshold:
            first = self._active_findings.get(
                "detection_dropout_rate",
                HealthFinding(signal="", severity=HealthSeverity.INFO,
                              message="", first_observed_frame=self._frame,
                              last_observed_frame=self._frame, recommended_action=""),
            ).first_observed_frame
            return HealthExplainer.explain_detection_dropout(rate, threshold, self._frame, first)
        return None

    def _check_depth_anomaly(self) -> Optional[HealthFinding]:
        """Check 3: Depth values clustering at clip bounds."""
        buf = self._buffers["depth_clip"]
        if len(buf) < self._win // 2:
            return None
        avg = sum(buf) / len(buf)
        threshold = float(self._cfg["depth_clip_boundary_max"])
        if avg > threshold:
            first = self._active_findings.get(
                "depth_estimation_anomaly",
                HealthFinding(signal="", severity=HealthSeverity.INFO,
                              message="", first_observed_frame=self._frame,
                              last_observed_frame=self._frame, recommended_action=""),
            ).first_observed_frame
            return HealthExplainer.explain_depth_anomaly(avg, threshold, self._frame, first)
        return None

    def _check_tracker_churn(self) -> Optional[HealthFinding]:
        """Check 4: New track ID spawn rate relative to detections."""
        buf = self._buffers["track_churn"]
        if len(buf) < self._win // 2:
            return None
        avg = sum(buf) / len(buf)
        threshold = float(self._cfg["tracker_churn_max"])
        if avg > threshold:
            first = self._active_findings.get(
                "tracker_churn_rate",
                HealthFinding(signal="", severity=HealthSeverity.INFO,
                              message="", first_observed_frame=self._frame,
                              last_observed_frame=self._frame, recommended_action=""),
            ).first_observed_frame
            return HealthExplainer.explain_tracker_churn(avg, threshold, self._frame, first)
        return None

    def _check_latency_trend(self) -> Optional[HealthFinding]:
        """Check 5: Rolling per-frame processing latency trend."""
        buf = self._buffers["frame_latency_ms"]
        if len(buf) < self._win // 2:
            return None
        avg = sum(buf) / len(buf)
        threshold = float(self._cfg["latency_max_ms"])
        if avg > threshold:
            first = self._active_findings.get(
                "frame_latency_trend",
                HealthFinding(signal="", severity=HealthSeverity.INFO,
                              message="", first_observed_frame=self._frame,
                              last_observed_frame=self._frame, recommended_action=""),
            ).first_observed_frame
            return HealthExplainer.explain_latency_trend(avg, threshold, self._frame, first)
        return None

    def _check_serial_health(self) -> Optional[HealthFinding]:
        """Check 6: Cumulative serial error / watchdog trigger count."""
        buf = self._buffers["serial_errors"]
        if not buf:
            return None
        current_total = buf[-1]
        threshold = float(self._cfg["serial_error_max"])
        if current_total >= threshold:
            first = self._active_findings.get(
                "serial_actuation_health",
                HealthFinding(signal="", severity=HealthSeverity.INFO,
                              message="", first_observed_frame=self._frame,
                              last_observed_frame=self._frame, recommended_action=""),
            ).first_observed_frame
            return HealthExplainer.explain_serial_health(
                int(current_total), int(threshold), self._frame, first
            )
        return None

    def _check_weather_fallback(self) -> Optional[HealthFinding]:
        """Check 7: Rate of frames using rule-based weather fallback."""
        buf = self._buffers["weather_fallback"]
        if len(buf) < self._win // 2:
            return None
        rate = sum(buf) / len(buf)
        threshold = float(self._cfg["weather_fallback_max"])
        if rate > threshold:
            first = self._active_findings.get(
                "weather_classifier_fallback",
                HealthFinding(signal="", severity=HealthSeverity.INFO,
                              message="", first_observed_frame=self._frame,
                              last_observed_frame=self._frame, recommended_action=""),
            ).first_observed_frame
            return HealthExplainer.explain_weather_fallback(rate, threshold, self._frame, first)
        return None

    def _check_beam_oscillation(self) -> Optional[HealthFinding]:
        """Check 8: Rate of beam state transitions within the window."""
        buf = self._buffers["beam_transitions"]
        if len(buf) < self._win // 2:
            return None
        rate = sum(buf) / len(buf)
        threshold = float(self._cfg["beam_oscillation_max"])
        if rate > threshold:
            first = self._active_findings.get(
                "beam_oscillation",
                HealthFinding(signal="", severity=HealthSeverity.INFO,
                              message="", first_observed_frame=self._frame,
                              last_observed_frame=self._frame, recommended_action=""),
            ).first_observed_frame
            return HealthExplainer.explain_beam_oscillation(rate, threshold, self._frame, first)
        return None

    def _check_attribution_gap(self) -> Optional[HealthFinding]:
        """Check 9: Frequency of high-risk zones with low attribution confidence."""
        buf = self._buffers["attribution_gap"]
        if len(buf) < self._win // 2:
            return None
        rate = sum(buf) / len(buf)
        threshold = float(self._cfg["explainability_gap_max"])
        if rate > threshold:
            first = self._active_findings.get(
                "explainability_confidence_gap",
                HealthFinding(signal="", severity=HealthSeverity.INFO,
                              message="", first_observed_frame=self._frame,
                              last_observed_frame=self._frame, recommended_action=""),
            ).first_observed_frame
            return HealthExplainer.explain_attribution_gap(rate, threshold, self._frame, first)
        return None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check_to_signal(check_name: str) -> str:
    """Map check method name → canonical signal name."""
    return {
        "_check_confidence_drift": "detection_confidence_drift",
        "_check_detection_dropout": "detection_dropout_rate",
        "_check_depth_anomaly": "depth_estimation_anomaly",
        "_check_tracker_churn": "tracker_churn_rate",
        "_check_latency_trend": "frame_latency_trend",
        "_check_serial_health": "serial_actuation_health",
        "_check_weather_fallback": "weather_classifier_fallback",
        "_check_beam_oscillation": "beam_oscillation",
        "_check_attribution_gap": "explainability_confidence_gap",
    }.get(check_name, check_name)


# ---------------------------------------------------------------------------
# Module-level singleton for lightweight import-and-use pattern
# ---------------------------------------------------------------------------
_default_monitor: Optional[HealthMonitor] = None


def get_monitor() -> HealthMonitor:
    """Return the module-level default HealthMonitor (creates it on first call)."""
    global _default_monitor
    if _default_monitor is None:
        _default_monitor = HealthMonitor()
    return _default_monitor


def record(stage: str, metrics: dict[str, Any]) -> None:
    """Module-level shortcut: ``diagnostics.record('detector', {...})``."""
    get_monitor().record(stage, metrics)
