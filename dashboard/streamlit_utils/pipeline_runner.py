"""
pipeline_runner.py — single-frame execution wrapper for the ADB pipeline.

Handles model loading (cached by Streamlit) and graceful fallback to
mock/synthetic mode if real weights or packages are unavailable.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is importable when run from the dashboard/ sub-folder
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from diagnostics.health_monitor import HealthMonitor
    from diagnostics.severity import HealthFinding
    _HAS_DIAGNOSTICS = True
except ImportError:
    _HAS_DIAGNOSTICS = False


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionInfo:
    """Lightweight info bundle for a single detection returned to the UI."""
    bbox: tuple[int, int, int, int]
    cls: int
    cls_name: str
    conf: float
    track_id: int
    distance_m: float
    lane_position: str
    center: tuple[int, int]
    risk_score: float
    zone_idx: int
    feature_contributions: dict[str, float] = field(default_factory=dict)


@dataclass
class FrameResult:
    """Everything the UI needs after one frame is processed."""
    frame: np.ndarray                    # Original (possibly undistorted) BGR frame
    detections: list[DetectionInfo]
    zone_risks: list[float]              # Per-zone risk [0-100], length = n_zones
    zone_brightness: list[int]           # Per-zone PWM brightness [0-255]
    beam_mode: str                       # BeamMode.value string
    weather: str                         # Weather class name
    n_zones: int
    processing_time_ms: float
    mock_flags: dict[str, bool]          # Which components are mocked
    ego_speed_kmh: float = 65.0
    # --- System-health diagnostics (separate from glare-risk explainability) ---
    health_findings: list[Any] = field(default_factory=list)   # list[HealthFinding]
    health_histories: dict[str, list[float]] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# PipelineRunner
# ──────────────────────────────────────────────────────────────────────────────

class PipelineRunner:
    """
    Wraps the full ADB perception → fusion → decision chain for a single frame.

    Args:
        mock: Force all perception modules into mock/synthetic mode.
        n_zones_override: If provided, override the zone count from config.
        ego_speed_kmh: Simulated ego vehicle speed (no GPS in demo mode).
    """

    def __init__(
        self,
        mock: bool = True,
        n_zones_override: int | None = None,
        ego_speed_kmh: float = 65.0,
    ) -> None:
        self._mock = mock
        self._ego_speed_kmh = ego_speed_kmh
        self._mock_flags: dict[str, bool] = {}

        # --- Detector ---
        try:
            from perception.detector import Detector
            self._detector = Detector(mock=mock)
            self._mock_flags["detector"] = self._detector._mock
        except Exception as exc:
            from perception.detector import Detector
            print(f"[PipelineRunner] Detector init warning: {exc}")
            self._detector = Detector(mock=True)
            self._mock_flags["detector"] = True

        # --- Depth Estimator ---
        try:
            from perception.depth import DepthEstimator
            self._depth = DepthEstimator(mock=mock)
            self._mock_flags["depth"] = self._depth._mock
        except Exception as exc:
            from perception.depth import DepthEstimator
            print(f"[PipelineRunner] DepthEstimator init warning: {exc}")
            self._depth = DepthEstimator(mock=True)
            self._mock_flags["depth"] = True

        # --- Tracker ---
        from perception.tracker import ObjectTracker
        self._tracker = ObjectTracker()
        self._mock_flags["tracker"] = False  # tracker is always real (no weights needed)

        # --- Lane Detector ---
        try:
            from perception.lane_detector import LaneDetector
            self._lane_detector = LaneDetector(mock=mock)
            self._mock_flags["lane_detector"] = mock
        except Exception as exc:
            from perception.lane_detector import LaneDetector
            print(f"[PipelineRunner] LaneDetector init warning: {exc}")
            self._lane_detector = LaneDetector(mock=True)
            self._mock_flags["lane_detector"] = True

        # --- Weather Classifier ---
        try:
            from perception.weather_classifier import WeatherClassifier
            self._weather = WeatherClassifier(mock=mock)
            self._mock_flags["weather"] = mock
        except Exception as exc:
            from perception.weather_classifier import WeatherClassifier
            print(f"[PipelineRunner] WeatherClassifier init warning: {exc}")
            self._weather = WeatherClassifier(mock=True)
            self._mock_flags["weather"] = True

        # --- Glare Risk Model (always heuristic, no weights needed) ---
        from fusion.glare_risk_model import GlareRiskModel, GlareFeatureVector
        self._glare_model = GlareRiskModel()
        self._GlareFeatureVector = GlareFeatureVector
        self._mock_flags["glare_model"] = False  # heuristic, always real

        # --- Zone Mapper ---
        from decision.zone_mapper import ZoneMapper
        self._zone_mapper = ZoneMapper()
        if n_zones_override and n_zones_override != self._zone_mapper.zone_count:
            # Patch the zone count dynamically without touching config files
            self._zone_mapper._n_zones = n_zones_override
            import numpy as _np
            self._zone_mapper._boundaries = list(
                _np.linspace(-40, 40, n_zones_override + 1)
            )
        self._mock_flags["zone_mapper"] = False

        # --- Beam State Machine ---
        from decision.beam_state_machine import BeamStateMachine
        self._beam_sm = BeamStateMachine()
        self._mock_flags["beam_sm"] = False

        # --- Serial Bridge (ALWAYS mocked — no real hardware in this app) ---
        from actuation.serial_bridge import SerialBridge
        self._bridge = SerialBridge(port="MOCK", mock=True)
        self._mock_flags["serial"] = True  # always

        # --- Health Monitor (shared singleton) ---
        if _HAS_DIAGNOSTICS:
            self._health_monitor = HealthMonitor()
        else:
            self._health_monitor = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def zone_count(self) -> int:
        return self._zone_mapper.zone_count

    @property
    def mock_flags(self) -> dict[str, bool]:
        return dict(self._mock_flags)

    def run_frame(self, frame: np.ndarray) -> FrameResult:
        """
        Execute one full pipeline pass on a BGR frame.

        Args:
            frame: OpenCV BGR image (H, W, 3).

        Returns:
            :class:`FrameResult` with all detections, risks, and beam decision.
        """
        t0 = time.perf_counter()

        # 1. Perception ───────────────────────────────────────────────────────
        detections = self._detector.detect(frame)
        depth_map = self._depth.estimate(frame)
        tracked_objects = self._tracker.update(detections, frame)
        lane_info = self._lane_detector.detect(frame)
        weather_cls, _weather_conf = self._weather.classify(frame)

        # 2. Fusion / Glare Risk ──────────────────────────────────────────────
        frame_w = frame.shape[1]
        per_object_risks: list[tuple[int, float]] = []  # (center_x, risk)
        detection_infos: list[DetectionInfo] = []

        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.bbox
            center_x = (x1 + x2) // 2

            # Estimate distance
            dist = self._depth.sample_depth(depth_map, obj.bbox)

            # Lane heuristic (oncoming = left half of frame)
            lane_pos = "oncoming" if center_x < frame_w // 2 else "adjacent_lane"

            rel_speed = float(getattr(obj, "velocity", (0.0, 0.0))[0]) * 0.5

            fv = self._GlareFeatureVector(
                object_type=obj.cls,
                distance=dist,
                relative_speed=max(rel_speed, 5.0),
                lane_position=lane_pos,
                vertical_angle=0.0,
                vehicle_height_class=1 if obj.cls in (5, 7) else 0,
                road_curvature=getattr(lane_info, "curvature_radius", 500.0),
                road_slope=0.0,
                weather_class=int(weather_cls),
                ambient_light=0.05,
                time_of_day=22.0,
            )
            risk_score = self._glare_model.predict(fv)
            zone_idx = self._zone_mapper.object_to_zone(center_x, frame_w)
            contribs = self._glare_model.contributions(fv)

            per_object_risks.append((center_x, risk_score))

            detection_infos.append(DetectionInfo(
                bbox=obj.bbox,
                cls=obj.cls,
                cls_name=obj.cls_name,
                conf=obj.conf,
                track_id=obj.track_id,
                distance_m=dist,
                lane_position=lane_pos,
                center=(center_x, (y1 + y2) // 2),
                risk_score=risk_score,
                zone_idx=zone_idx,
                feature_contributions=contribs if isinstance(contribs, dict) else {},
            ))

        # 3. Decision ─────────────────────────────────────────────────────────
        zone_risks = self._zone_mapper.aggregate_risks(per_object_risks)
        zone_pwm = self._zone_mapper.compute_brightness(zone_risks)

        beam_mode = self._beam_sm.update(
            zone_risks=zone_risks,
            ego_speed_kmh=self._ego_speed_kmh,
            weather=weather_cls.name.lower(),
        )

        # 4. Actuation (always mocked) ─────────────────────────────────────
        self._bridge.send_pwm(zone_pwm)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 5. Health Monitor — record latency, attribution gaps, then evaluate —
        health_findings: list = []
        health_histories: dict[str, list[float]] = {}

        if self._health_monitor is not None:
            # Record per-frame latency
            self._health_monitor.record("pipeline", {"latency_ms": elapsed_ms})

            # Record explainability attribution-gap signal
            # Gap = any high-risk decision whose top contribution magnitude < 0.05
            for info in detection_infos:
                if info.risk_score > 60.0 and info.feature_contributions:
                    max_attr = max(abs(v) for v in info.feature_contributions.values())
                    has_gap = max_attr < 0.05
                    self._health_monitor.record("fusion", {"attribution_gap": has_gap})

            self._health_monitor.tick()
            health_findings = self._health_monitor.evaluate()
            health_histories = self._health_monitor.get_all_histories()

        return FrameResult(
            frame=frame,
            detections=detection_infos,
            zone_risks=zone_risks,
            zone_brightness=zone_pwm,
            beam_mode=beam_mode.value,
            weather=weather_cls.name.lower(),
            n_zones=self._zone_mapper.zone_count,
            processing_time_ms=elapsed_ms,
            mock_flags=dict(self._mock_flags),
            ego_speed_kmh=self._ego_speed_kmh,
            health_findings=health_findings,
            health_histories=health_histories,
        )
