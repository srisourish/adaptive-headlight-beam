"""
tests/unit/test_health_diagnostics.py — Unit tests for System Health & Predictive Diagnostics.

Covers:
  - Normal operation: no findings under baseline signals
  - Degraded signal: one test per check that verifies the correct finding fires
    at the expected severity when the signal is pushed past threshold
  - Deduplication: a persistent condition shows one finding with updating last_observed_frame
  - Severity classification: worst_severity helper
  - HealthFinding dataclass helpers (duration_frames, to_dict)

Each degraded-signal test pushes the buffer to (window_size // 2 + 1) entries so
the check has enough data to evaluate, matching the ``len(buf) < self._win // 2``
guard in health_monitor.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.health_monitor import HealthMonitor
from diagnostics.severity import HealthFinding, HealthSeverity, worst_severity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_monitor(window: int = 20) -> HealthMonitor:
    """Create a HealthMonitor with a small window for fast tests."""
    m = HealthMonitor(window_size=window)
    return m


def fill_buffer(monitor: HealthMonitor, stage: str, metrics_list: list) -> None:
    """Push a list of metric dicts into the monitor, advancing frame each time."""
    for m in metrics_list:
        monitor.record(stage, m)
        monitor.tick()


# ---------------------------------------------------------------------------
# Test 1: Normal Operation — No Findings
# ---------------------------------------------------------------------------

class TestNormalOperation:
    """Under healthy baseline signals, evaluate() should return no findings."""

    def test_no_findings_on_healthy_detector(self):
        mon = make_monitor()
        # Healthy: high confidence, no dropouts
        for _ in range(15):
            mon.record("detector", {"conf": 0.85, "n_detections": 3})
            mon.tick()
        findings = mon.evaluate()
        assert findings == [], f"Expected no findings, got: {findings}"

    def test_no_findings_on_healthy_depth(self):
        mon = make_monitor()
        for _ in range(15):
            mon.record("depth", {"clip_fraction": 0.02})
            mon.tick()
        findings = mon.evaluate()
        assert findings == []

    def test_no_findings_on_healthy_tracker(self):
        mon = make_monitor()
        for _ in range(15):
            mon.record("tracker", {"n_new_ids": 0, "n_detections": 3})
            mon.tick()
        findings = mon.evaluate()
        assert findings == []

    def test_no_findings_on_healthy_latency(self):
        mon = make_monitor()
        for _ in range(15):
            mon.record("pipeline", {"latency_ms": 25.0})
            mon.tick()
        findings = mon.evaluate()
        assert findings == []

    def test_no_findings_on_healthy_serial(self):
        mon = make_monitor()
        for _ in range(15):
            mon.record("serial_bridge", {"error_count": 0})
            mon.tick()
        findings = mon.evaluate()
        assert findings == []

    def test_no_findings_on_healthy_beam(self):
        mon = make_monitor()
        for _ in range(15):
            mon.record("beam_sm", {"transitioned": False})
            mon.tick()
        findings = mon.evaluate()
        assert findings == []

    def test_no_findings_insufficient_data(self):
        """With fewer than window/2 samples, checks should not fire."""
        mon = make_monitor(window=40)
        # Only 3 samples — not enough for any check
        for _ in range(3):
            mon.record("detector", {"conf": 0.10, "n_detections": 0})
            mon.tick()
        findings = mon.evaluate()
        assert findings == [], "Should not fire with insufficient data"


# ---------------------------------------------------------------------------
# Test 2: Check 1 — Detection Confidence Drift
# ---------------------------------------------------------------------------

class TestDetectionConfidenceDrift:
    def test_warning_fires_below_threshold(self):
        mon = make_monitor(window=20)
        # Default threshold is 0.55; inject 0.30 → well below
        for _ in range(12):
            mon.record("detector", {"conf": 0.30, "n_detections": 2})
            mon.tick()
        findings = mon.evaluate()
        signals = [f.signal for f in findings]
        assert "detection_confidence_drift" in signals

    def test_severity_is_warning_just_below_threshold(self):
        mon = make_monitor(window=20)
        for _ in range(12):
            mon.record("detector", {"conf": 0.48, "n_detections": 2})
            mon.tick()
        findings = {f.signal: f for f in mon.evaluate()}
        f = findings.get("detection_confidence_drift")
        assert f is not None
        assert f.severity in (HealthSeverity.WARNING, HealthSeverity.CRITICAL)

    def test_critical_fires_far_below_threshold(self):
        mon = make_monitor(window=20)
        # 75% of 0.55 threshold = 0.4125; 0.20 is well below → CRITICAL
        for _ in range(12):
            mon.record("detector", {"conf": 0.20, "n_detections": 2})
            mon.tick()
        findings = {f.signal: f for f in mon.evaluate()}
        f = findings.get("detection_confidence_drift")
        assert f is not None
        assert f.severity == HealthSeverity.CRITICAL

    def test_no_finding_above_threshold(self):
        mon = make_monitor(window=20)
        for _ in range(12):
            mon.record("detector", {"conf": 0.80, "n_detections": 2})
            mon.tick()
        signals = [f.signal for f in mon.evaluate()]
        assert "detection_confidence_drift" not in signals


# ---------------------------------------------------------------------------
# Test 3: Check 2 — Detection Dropout Rate
# ---------------------------------------------------------------------------

class TestDetectionDropout:
    def test_warning_fires_on_high_dropout(self):
        mon = make_monitor(window=20)
        # 70% dropouts > default threshold 0.40
        for i in range(12):
            zero = i % 10 >= 3  # 7 out of 10 are zero-detection
            mon.record("detector", {"conf": 0.0 if zero else 0.7, "n_detections": 0 if zero else 2})
            mon.tick()
        findings = [f.signal for f in mon.evaluate()]
        assert "detection_dropout_rate" in findings

    def test_no_finding_with_low_dropout(self):
        mon = make_monitor(window=20)
        for _ in range(12):
            mon.record("detector", {"conf": 0.8, "n_detections": 3})
            mon.tick()
        signals = [f.signal for f in mon.evaluate()]
        assert "detection_dropout_rate" not in signals


# ---------------------------------------------------------------------------
# Test 4: Check 3 — Depth Estimation Anomaly
# ---------------------------------------------------------------------------

class TestDepthAnomaly:
    def test_warning_fires_on_high_clip_fraction(self):
        mon = make_monitor(window=20)
        # 60% at clip bounds > default threshold 0.35
        for _ in range(12):
            mon.record("depth", {"clip_fraction": 0.60})
            mon.tick()
        signals = [f.signal for f in mon.evaluate()]
        assert "depth_estimation_anomaly" in signals

    def test_no_finding_on_normal_depth(self):
        mon = make_monitor(window=20)
        for _ in range(12):
            mon.record("depth", {"clip_fraction": 0.05})
            mon.tick()
        signals = [f.signal for f in mon.evaluate()]
        assert "depth_estimation_anomaly" not in signals


# ---------------------------------------------------------------------------
# Test 5: Check 4 — Tracker Churn Rate
# ---------------------------------------------------------------------------

class TestTrackerChurn:
    def test_warning_fires_on_high_churn(self):
        mon = make_monitor(window=20)
        # Churn rate 0.9 > default threshold 0.60
        for _ in range(12):
            mon.record("tracker", {"n_new_ids": 3, "n_detections": 3})
            mon.tick()
        signals = [f.signal for f in mon.evaluate()]
        assert "tracker_churn_rate" in signals

    def test_no_finding_on_stable_tracking(self):
        mon = make_monitor(window=20)
        for _ in range(12):
            mon.record("tracker", {"n_new_ids": 0, "n_detections": 3})
            mon.tick()
        signals = [f.signal for f in mon.evaluate()]
        assert "tracker_churn_rate" not in signals


# ---------------------------------------------------------------------------
# Test 6: Check 5 — Frame Latency Trend
# ---------------------------------------------------------------------------

class TestLatencyTrend:
    def test_warning_fires_on_high_latency(self):
        mon = make_monitor(window=20)
        # 200ms > default threshold 120ms
        for _ in range(12):
            mon.record("pipeline", {"latency_ms": 200.0})
            mon.tick()
        signals = [f.signal for f in mon.evaluate()]
        assert "frame_latency_trend" in signals

    def test_critical_fires_on_very_high_latency(self):
        mon = make_monitor(window=20)
        # 1.5x threshold = 180ms → CRITICAL; use 300ms
        for _ in range(12):
            mon.record("pipeline", {"latency_ms": 300.0})
            mon.tick()
        findings = {f.signal: f for f in mon.evaluate()}
        f = findings.get("frame_latency_trend")
        assert f is not None
        assert f.severity == HealthSeverity.CRITICAL

    def test_no_finding_on_normal_latency(self):
        mon = make_monitor(window=20)
        for _ in range(12):
            mon.record("pipeline", {"latency_ms": 20.0})
            mon.tick()
        signals = [f.signal for f in mon.evaluate()]
        assert "frame_latency_trend" not in signals


# ---------------------------------------------------------------------------
# Test 7: Check 6 — Serial / Actuation Health
# ---------------------------------------------------------------------------

class TestSerialHealth:
    def test_warning_fires_on_errors(self):
        mon = make_monitor(window=20)
        # Inject 2 errors → above threshold of 1
        mon.record("serial_bridge", {"error_count": 2})
        mon.tick()
        mon.evaluate()  # let the total accumulate
        signals = [f.signal for f in mon.evaluate()]
        assert "serial_actuation_health" in signals

    def test_no_finding_on_zero_errors(self):
        mon = make_monitor(window=20)
        for _ in range(5):
            mon.record("serial_bridge", {"error_count": 0})
            mon.tick()
        signals = [f.signal for f in mon.evaluate()]
        assert "serial_actuation_health" not in signals


# ---------------------------------------------------------------------------
# Test 8: Check 7 — Weather Classifier Fallback
# ---------------------------------------------------------------------------

class TestWeatherFallback:
    def test_warning_fires_on_high_fallback_rate(self):
        mon = make_monitor(window=20)
        # 80% fallback rate > default 0.50
        for i in range(12):
            mon.record("weather", {"used_fallback": i % 10 < 8})
            mon.tick()
        signals = [f.signal for f in mon.evaluate()]
        assert "weather_classifier_fallback" in signals

    def test_no_finding_when_cnn_used(self):
        mon = make_monitor(window=20)
        for _ in range(12):
            mon.record("weather", {"used_fallback": False})
            mon.tick()
        signals = [f.signal for f in mon.evaluate()]
        assert "weather_classifier_fallback" not in signals


# ---------------------------------------------------------------------------
# Test 9: Check 8 — Beam State Machine Oscillation
# ---------------------------------------------------------------------------

class TestBeamOscillation:
    def test_warning_fires_on_high_transition_rate(self):
        mon = make_monitor(window=20)
        # Every frame transitions → rate 1.0 >> default 0.15
        for _ in range(12):
            mon.record("beam_sm", {"transitioned": True})
            mon.tick()
        signals = [f.signal for f in mon.evaluate()]
        assert "beam_oscillation" in signals

    def test_no_finding_on_stable_beam(self):
        mon = make_monitor(window=20)
        for _ in range(12):
            mon.record("beam_sm", {"transitioned": False})
            mon.tick()
        signals = [f.signal for f in mon.evaluate()]
        assert "beam_oscillation" not in signals


# ---------------------------------------------------------------------------
# Test 10: Check 9 — Explainability Attribution Gap
# ---------------------------------------------------------------------------

class TestAttributionGap:
    def test_warning_fires_on_frequent_gap(self):
        mon = make_monitor(window=20)
        # 50% gap rate > default 0.30
        for i in range(12):
            mon.record("fusion", {"attribution_gap": i % 2 == 0})
            mon.tick()
        signals = [f.signal for f in mon.evaluate()]
        assert "explainability_confidence_gap" in signals

    def test_no_finding_on_healthy_attribution(self):
        mon = make_monitor(window=20)
        for _ in range(12):
            mon.record("fusion", {"attribution_gap": False})
            mon.tick()
        signals = [f.signal for f in mon.evaluate()]
        assert "explainability_confidence_gap" not in signals


# ---------------------------------------------------------------------------
# Test 11: Deduplication — Persistent condition shows one finding, updating last_observed_frame
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_persistent_finding_not_duplicated(self):
        """A persistent condition should produce exactly one finding entry."""
        mon = make_monitor(window=20)
        # Fill buffer with bad confidence
        for _ in range(12):
            mon.record("detector", {"conf": 0.20, "n_detections": 2})
            mon.tick()
        findings_1 = mon.evaluate()
        conf_findings_1 = [f for f in findings_1 if f.signal == "detection_confidence_drift"]
        assert len(conf_findings_1) == 1, "Should be exactly one finding for the signal"
        first_last_frame = conf_findings_1[0].last_observed_frame

        # Continue for more frames — still only one finding, but last_observed updated
        for _ in range(6):
            mon.record("detector", {"conf": 0.20, "n_detections": 2})
            mon.tick()
        findings_2 = mon.evaluate()
        conf_findings_2 = [f for f in findings_2 if f.signal == "detection_confidence_drift"]
        assert len(conf_findings_2) == 1, "Must not duplicate — deduplication required"
        assert conf_findings_2[0].last_observed_frame >= first_last_frame, \
            "last_observed_frame should advance or stay equal with continued bad signal"

    def test_finding_clears_when_signal_normalises(self):
        """Once a degraded signal recovers, the finding should be resolved."""
        mon = make_monitor(window=20)
        # Trigger finding
        for _ in range(12):
            mon.record("detector", {"conf": 0.20, "n_detections": 2})
            mon.tick()
        findings_bad = mon.evaluate()
        assert any(f.signal == "detection_confidence_drift" for f in findings_bad)

        # Recover signal — fill buffer with good values
        for _ in range(20):
            mon.record("detector", {"conf": 0.90, "n_detections": 3})
            mon.tick()
        findings_good = mon.evaluate()
        # Should no longer appear as active
        assert all(f.signal != "detection_confidence_drift" for f in findings_good), \
            "Finding should resolve once signal recovers"


# ---------------------------------------------------------------------------
# Test 12: Severity classification helpers
# ---------------------------------------------------------------------------

class TestSeverityHelpers:
    def test_worst_severity_returns_critical(self):
        findings = [
            HealthFinding("s1", HealthSeverity.INFO, "msg", 0, 0, "action"),
            HealthFinding("s2", HealthSeverity.CRITICAL, "msg", 0, 0, "action"),
            HealthFinding("s3", HealthSeverity.WARNING, "msg", 0, 0, "action"),
        ]
        assert worst_severity(findings) == HealthSeverity.CRITICAL

    def test_worst_severity_empty_list_returns_none(self):
        assert worst_severity([]) is None

    def test_worst_severity_all_info(self):
        findings = [
            HealthFinding("s1", HealthSeverity.INFO, "msg", 0, 0, "action"),
        ]
        assert worst_severity(findings) == HealthSeverity.INFO

    def test_severity_ordering(self):
        assert HealthSeverity.CRITICAL > HealthSeverity.WARNING
        assert HealthSeverity.WARNING > HealthSeverity.INFO
        assert HealthSeverity.INFO < HealthSeverity.CRITICAL

    def test_health_finding_duration(self):
        f = HealthFinding("sig", HealthSeverity.WARNING, "msg", first_observed_frame=10,
                          last_observed_frame=25, recommended_action="act")
        assert f.duration_frames == 15

    def test_health_finding_to_dict(self):
        f = HealthFinding("sig", HealthSeverity.WARNING, "msg", first_observed_frame=10,
                          last_observed_frame=25, recommended_action="act",
                          metric_name="val", metric_value=0.123)
        d = f.to_dict()
        assert d["signal"] == "sig"
        assert d["severity"] == "WARNING"
        assert d["duration_frames"] == 15
        assert abs(d["metric_value"] - 0.123) < 0.001
        assert not d["is_resolved"]


# ---------------------------------------------------------------------------
# Test 13: get_metric_history returns data
# ---------------------------------------------------------------------------

class TestMetricHistory:
    def test_history_populated_after_records(self):
        mon = make_monitor(window=20)
        for i in range(5):
            mon.record("detector", {"conf": 0.6 + i * 0.01, "n_detections": 2})
            mon.tick()
        hist = mon.get_metric_history("detection_conf")
        assert len(hist) == 5
        assert all(isinstance(v, float) for v in hist)

    def test_history_empty_for_unknown_key(self):
        mon = make_monitor()
        assert mon.get_metric_history("nonexistent_signal") == []
