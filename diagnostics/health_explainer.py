"""
diagnostics/health_explainer.py — Plain-language diagnostic report generator.

Translates raw metric states and triggered check identifiers into human-readable
findings using the same transparency philosophy as fusion/explainability.py, but
applied to the *health of the pipeline itself* rather than driving decisions.

Design notes:
- Each check gets a dedicated _explain_* method so message templates are
  co-located with the check logic they describe.
- Messages avoid alarming language. They state what was observed, give a
  likely cause, and end with a concrete next step.
- Severity levels follow: INFO → informational, WARNING → investigate soon,
  CRITICAL → investigate before relying on system output.
"""

from __future__ import annotations

from diagnostics.severity import HealthFinding, HealthSeverity


class HealthExplainer:
    """Converts raw diagnostic check results into HealthFinding records.

    Each ``explain_*`` method accepts the triggering metric value and a frame
    index, and returns a populated ``HealthFinding`` ready for the dashboard.
    """

    # ------------------------------------------------------------------
    # 1. Detection confidence drift
    # ------------------------------------------------------------------
    @staticmethod
    def explain_confidence_drift(
        rolling_avg_conf: float,
        threshold: float,
        frame: int,
        first_frame: int,
    ) -> HealthFinding:
        severity = (
            HealthSeverity.CRITICAL
            if rolling_avg_conf < threshold * 0.75
            else HealthSeverity.WARNING
        )
        msg = (
            f"Rolling detection confidence has dropped to {rolling_avg_conf:.2f} "
            f"(threshold: {threshold:.2f}) over the last several frames. "
            "This may indicate camera obstruction, poor lighting, lens contamination, "
            "or a mismatch between the YOLO checkpoint and the current scene."
        )
        action = (
            "Check the camera lens for smudges or physical obstruction. "
            "Verify lighting conditions and confirm the correct model checkpoint is loaded."
        )
        return HealthFinding(
            signal="detection_confidence_drift",
            severity=severity,
            message=msg,
            first_observed_frame=first_frame,
            last_observed_frame=frame,
            recommended_action=action,
            metric_name="Rolling Avg Confidence",
            metric_value=rolling_avg_conf,
        )

    # ------------------------------------------------------------------
    # 2. Detection dropout rate
    # ------------------------------------------------------------------
    @staticmethod
    def explain_detection_dropout(
        dropout_rate: float,
        threshold: float,
        frame: int,
        first_frame: int,
    ) -> HealthFinding:
        severity = (
            HealthSeverity.CRITICAL
            if dropout_rate > min(threshold * 1.5, 0.9)
            else HealthSeverity.WARNING
        )
        msg = (
            f"{dropout_rate * 100:.0f}% of recent frames returned zero detections "
            f"(threshold: {threshold * 100:.0f}%). "
            "A sudden increase in empty frames often signals a camera feed interruption, "
            "severe exposure problem, or the blob-fallback detector engaging unexpectedly "
            "on scenes it cannot handle."
        )
        action = (
            "Inspect the camera feed for connection issues or extreme over-/under-exposure. "
            "If using live video, check cable seating and capture device driver status."
        )
        return HealthFinding(
            signal="detection_dropout_rate",
            severity=severity,
            message=msg,
            first_observed_frame=first_frame,
            last_observed_frame=frame,
            recommended_action=action,
            metric_name="Dropout Rate",
            metric_value=dropout_rate,
        )

    # ------------------------------------------------------------------
    # 3. Depth estimation anomalies
    # ------------------------------------------------------------------
    @staticmethod
    def explain_depth_anomaly(
        clip_fraction: float,
        threshold: float,
        frame: int,
        first_frame: int,
    ) -> HealthFinding:
        severity = (
            HealthSeverity.CRITICAL
            if clip_fraction > min(threshold * 1.6, 0.9)
            else HealthSeverity.WARNING
        )
        msg = (
            f"{clip_fraction * 100:.0f}% of recent depth samples are at the clip boundary "
            f"(≤0.1 m or ≥200 m), above the threshold of {threshold * 100:.0f}%. "
            "Depth values clustering at clip limits typically indicate miscalibration of "
            "the depth scale factor S, or a silent MiDaS checkpoint fallback returning "
            "near-uniform outputs."
        )
        action = (
            "Review the 'depth_scale_factor' in camera_calib.yaml and re-run calibration. "
            "Verify that the MiDaS checkpoint path exists and loads without errors."
        )
        return HealthFinding(
            signal="depth_estimation_anomaly",
            severity=severity,
            message=msg,
            first_observed_frame=first_frame,
            last_observed_frame=frame,
            recommended_action=action,
            metric_name="Depth Clip Fraction",
            metric_value=clip_fraction,
        )

    # ------------------------------------------------------------------
    # 4. Tracker churn rate
    # ------------------------------------------------------------------
    @staticmethod
    def explain_tracker_churn(
        churn_rate: float,
        threshold: float,
        frame: int,
        first_frame: int,
    ) -> HealthFinding:
        severity = (
            HealthSeverity.CRITICAL
            if churn_rate > min(threshold * 1.5, 1.0)
            else HealthSeverity.WARNING
        )
        msg = (
            f"Tracker is spawning new IDs at a rate of {churn_rate:.2f} per frame "
            f"(threshold: {threshold:.2f}), indicating tracks are not persisting. "
            "High churn suggests detection boxes are too noisy or jittery, or the IOU "
            "matching threshold is configured too strictly for the current scene density."
        )
        action = (
            "Lower detection noise by increasing min_confidence in thresholds.yaml. "
            "Alternatively, widen the tracker's IOU threshold slightly "
            "(ObjectTracker iou_threshold parameter)."
        )
        return HealthFinding(
            signal="tracker_churn_rate",
            severity=severity,
            message=msg,
            first_observed_frame=first_frame,
            last_observed_frame=frame,
            recommended_action=action,
            metric_name="Track Churn Rate",
            metric_value=churn_rate,
        )

    # ------------------------------------------------------------------
    # 5. Frame processing latency trend
    # ------------------------------------------------------------------
    @staticmethod
    def explain_latency_trend(
        rolling_avg_ms: float,
        threshold_ms: float,
        frame: int,
        first_frame: int,
    ) -> HealthFinding:
        severity = (
            HealthSeverity.CRITICAL
            if rolling_avg_ms > threshold_ms * 1.5
            else HealthSeverity.WARNING
        )
        msg = (
            f"Rolling average frame latency is {rolling_avg_ms:.1f} ms "
            f"(threshold: {threshold_ms:.0f} ms). "
            "Rising latency within a session often signals thermal throttling on the "
            "compute device, a memory leak in one of the perception stages, or resource "
            "contention from another process."
        )
        action = (
            "Check CPU/GPU temperature and throttling status. "
            "Monitor memory usage growth. Consider restarting the pipeline if latency "
            "continues to rise without external cause."
        )
        return HealthFinding(
            signal="frame_latency_trend",
            severity=severity,
            message=msg,
            first_observed_frame=first_frame,
            last_observed_frame=frame,
            recommended_action=action,
            metric_name="Latency (ms)",
            metric_value=rolling_avg_ms,
        )

    # ------------------------------------------------------------------
    # 6. Serial / actuation health
    # ------------------------------------------------------------------
    @staticmethod
    def explain_serial_health(
        error_count: int,
        threshold: int,
        frame: int,
        first_frame: int,
    ) -> HealthFinding:
        severity = (
            HealthSeverity.CRITICAL
            if error_count >= threshold * 3
            else HealthSeverity.WARNING
        )
        msg = (
            f"{error_count} serial/actuation error event(s) detected "
            f"(threshold: {threshold}) since the session started. "
            "This includes watchdog fail-safe triggers, dropped or malformed packets, "
            "and serial reconnect events. Likely causes: loose wiring, port contention, "
            "or a firmware crash-loop on the Arduino."
        )
        action = (
            "Check the USB/serial cable connection and port availability. "
            "Inspect Arduino firmware logs for crash or watchdog reset indicators. "
            "Verify no other process is holding the serial port open."
        )
        return HealthFinding(
            signal="serial_actuation_health",
            severity=severity,
            message=msg,
            first_observed_frame=first_frame,
            last_observed_frame=frame,
            recommended_action=action,
            metric_name="Serial Error Count",
            metric_value=float(error_count),
        )

    # ------------------------------------------------------------------
    # 7. Weather classifier fallback engagement
    # ------------------------------------------------------------------
    @staticmethod
    def explain_weather_fallback(
        fallback_rate: float,
        threshold: float,
        frame: int,
        first_frame: int,
    ) -> HealthFinding:
        severity = (
            HealthSeverity.CRITICAL
            if fallback_rate > min(threshold * 1.5, 1.0)
            else HealthSeverity.WARNING
        )
        msg = (
            f"The weather classifier is using the rule-based fallback on "
            f"{fallback_rate * 100:.0f}% of frames (threshold: {threshold * 100:.0f}%). "
            "This indicates the learned CNN model checkpoint may be missing, corrupted, "
            "or silently failing to load, forcing the heuristic Laplacian/HSV path."
        )
        action = (
            "Verify the CNN model checkpoint exists at the configured path in models/weights/. "
            "Re-run model training or restore from backup if the file is corrupted."
        )
        return HealthFinding(
            signal="weather_classifier_fallback",
            severity=severity,
            message=msg,
            first_observed_frame=first_frame,
            last_observed_frame=frame,
            recommended_action=action,
            metric_name="Fallback Rate",
            metric_value=fallback_rate,
        )

    # ------------------------------------------------------------------
    # 8. Beam state machine oscillation
    # ------------------------------------------------------------------
    @staticmethod
    def explain_beam_oscillation(
        transition_rate: float,
        threshold: float,
        frame: int,
        first_frame: int,
    ) -> HealthFinding:
        severity = (
            HealthSeverity.CRITICAL
            if transition_rate > threshold * 1.5
            else HealthSeverity.WARNING
        )
        msg = (
            f"The beam state machine is transitioning at {transition_rate:.2f} times/frame, "
            f"above the expected bound of {threshold:.2f}. "
            "Despite asymmetric hysteresis protection, rapid oscillation suggests the "
            "upstream risk score itself is oscillating — possibly due to unstable detections "
            "or a misconfigured hysteresis timing parameter."
        )
        action = (
            "Review 'transition_to_protect_s' and 'transition_to_restore_s' in thresholds.yaml. "
            "Inspect per-zone risk scores for high-frequency oscillation. "
            "Consider increasing the EMA smoothing alpha to stabilise the signal."
        )
        return HealthFinding(
            signal="beam_oscillation",
            severity=severity,
            message=msg,
            first_observed_frame=first_frame,
            last_observed_frame=frame,
            recommended_action=action,
            metric_name="Transition Rate",
            metric_value=transition_rate,
        )

    # ------------------------------------------------------------------
    # 9. Explainability confidence gaps
    # ------------------------------------------------------------------
    @staticmethod
    def explain_attribution_gap(
        gap_rate: float,
        threshold: float,
        frame: int,
        first_frame: int,
    ) -> HealthFinding:
        severity = (
            HealthSeverity.WARNING
            if gap_rate < threshold * 1.5
            else HealthSeverity.CRITICAL
        )
        msg = (
            f"{gap_rate * 100:.0f}% of high-risk zone decisions have unusually low "
            f"top-feature attribution confidence (threshold: {threshold * 100:.0f}%). "
            "This suggests the fusion layer feature vector may be partially corrupted, "
            "or a bug is causing near-zero contributions for all features on some inputs."
        )
        action = (
            "Inspect the GlareFeatureVector construction in pipeline_runner.py for "
            "NaN/zero inputs. Check that all fusion weights in thresholds.yaml sum "
            "to approximately 1.0 and none are zero."
        )
        return HealthFinding(
            signal="explainability_confidence_gap",
            severity=severity,
            message=msg,
            first_observed_frame=first_frame,
            last_observed_frame=frame,
            recommended_action=action,
            metric_name="Attribution Gap Rate",
            metric_value=gap_rate,
        )
