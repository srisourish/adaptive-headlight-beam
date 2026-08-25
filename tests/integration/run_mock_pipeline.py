"""
Integration runner for smart-adaptive-headlight.

Runs full end-to-end perception -> fusion -> decision -> actuation mock pipeline loop
on synthetic frames without hardware attached.
Prints per-zone risk and beam mode each frame.

Usage
-----
    python tests/integration/run_mock_pipeline.py --frames 50
    python tests/integration/run_mock_pipeline.py --frames 50 --degrade
        # --degrade: injects artificial signal degradation to verify
        #            health diagnostic checks fire.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from actuation.serial_bridge import SerialBridge
from decision.beam_state_machine import BeamStateMachine
from decision.zone_mapper import ZoneMapper
from fusion.glare_risk_model import GlareFeatureVector, GlareRiskModel
from perception.camera_capture import CameraCapture
from perception.depth import DepthEstimator
from perception.detector import Detector
from perception.lane_detector import LaneDetector
from perception.tracker import ObjectTracker
from perception.weather_classifier import WeatherClassifier

try:
    from diagnostics.health_monitor import HealthMonitor
    _HAS_DIAGNOSTICS = True
except ImportError:
    _HAS_DIAGNOSTICS = False


def run_pipeline(
    num_frames: int = 30,
    verbose: bool = True,
    source: str | None = None,
    degrade: bool = False,
) -> None:
    """Execute end-to-end adaptive headlight control loop.

    Args:
        degrade: If True, inject artificially degraded signals to exercise
                 the health diagnostic checks (confidence drift + serial error).
                 Acceptance criteria: at least one health finding should fire.
    """
    print("=" * 70)
    print("  SMART ADAPTIVE HEADLIGHT (ADB) -- INTEGRATION MOCK PIPELINE")
    if degrade:
        print("  [WARN] DEGRADED SCENARIO MODE: injecting low-confidence + serial error")
    print("=" * 70)

    use_mock = source is None
    # Allow "0", "1" etc. to mean webcam index instead of a file path
    if source is not None and source.isdigit():
        source = int(source)

    capture = CameraCapture(source=source, mock=use_mock, mock_size=(1280, 720))
    print(f"[Pipeline] Camera mode: {'MOCK (synthetic)' if capture.is_mock else f'LIVE (source={source})'}")
    detector = Detector(mock=True)
    depth_estimator = DepthEstimator(mock=True)
    tracker = ObjectTracker()
    lane_detector = LaneDetector(mock=True)
    weather_classifier = WeatherClassifier(mock=True)
    glare_model = GlareRiskModel()
    zone_mapper = ZoneMapper()
    state_machine = BeamStateMachine()
    serial_bridge = SerialBridge(port="MOCK", mock=True)

    # Health monitor (use module singleton so all stages share it)
    monitor = HealthMonitor() if _HAS_DIAGNOSTICS else None

    print("\nStarting pipeline processing loop...\n")
    start_time = time.time()

    for frame_idx in range(1, num_frames + 1):
        ret, frame = capture.read()
        if not ret or frame is None:
            break

        # 1. Perception
        detections = detector.detect(frame)
        depth_map = depth_estimator.estimate(frame)
        tracked_objects = tracker.update(detections, frame)
        lane_info = lane_detector.detect(frame)
        weather_cls, _ = weather_classifier.classify(frame)

        # 2. Fusion & Glare Risk calculation per tracked object
        per_object_risks: list[tuple[int, float]] = []  # (center_x, risk_score)
        for obj in tracked_objects:
            center_x = (obj.bbox[0] + obj.bbox[2]) // 2
            dist = getattr(obj, "distance", 35.0)
            rel_speed = getattr(obj, "relative_speed", 15.0)

            fv = GlareFeatureVector(
                object_type=obj.cls,
                distance=dist,
                relative_speed=rel_speed,
                lane_position="oncoming" if obj.cls in (2, 7) else "adjacent_lane",
                vertical_angle=0.0,
                vehicle_height_class=1,
                road_curvature=lane_info.curvature if hasattr(lane_info, "curvature") else 500.0,
                road_slope=0.0,
                weather_class=int(weather_cls),
                ambient_light=0.05,
                time_of_day=22.0,
            )
            risk_score = glare_model.predict(fv)
            per_object_risks.append((center_x, risk_score))

        # 3. Decision
        zone_risks = zone_mapper.aggregate_risks(per_object_risks)
        zone_pwm = zone_mapper.compute_brightness(zone_risks)

        # Simulated ego speed (km/h)
        ego_speed = 65.0
        beam_mode = state_machine.update(
            zone_risks=zone_risks, ego_speed_kmh=ego_speed, weather=weather_cls.name.lower()
        )

        # 4. Actuation
        serial_bridge.send_pwm(zone_pwm)

        # Inject degraded signals for testing (after normal pipeline run)
        if degrade and monitor:
            # Simulate low detection confidence → triggers Check 1
            monitor.record("detector", {"conf": 0.20, "n_detections": max(1, len(tracked_objects))})
            # Simulate high dropout rate → triggers Check 2
            monitor.record("detector", {"conf": 0.0, "n_detections": 0})
            # Simulate serial watchdog trigger on frame 5
            if frame_idx == 5:
                monitor.record("serial_bridge", {"error_count": 2})

        if monitor:
            monitor.record("pipeline", {"latency_ms": (time.time() - start_time) * 1000 / frame_idx})
            monitor.tick()

        # Formatted frame output log
        findings = monitor.evaluate() if monitor else []
        health_str = " | " + f"Health: {len(findings)} finding(s)" if findings else ""

        risks_str = ", ".join(f"Z{i}:{r:4.1f}%" for i, r in enumerate(zone_risks))
        pwm_str = ", ".join(f"Z{i}:{p}" for i, p in enumerate(zone_pwm))

        if verbose:
            print(
                f"[Frame {frame_idx:03d}/{num_frames:03d}] "
                f"Mode: {beam_mode.value:14s} | "
                f"Objs: {len(tracked_objects)} | "
                f"Per-Zone Risks: [{risks_str}]"
                f"{health_str}"
            )

    elapsed = time.time() - start_time
    fps = num_frames / max(elapsed, 0.001)

    print("\n" + "=" * 70)
    print(
        f"Pipeline complete! Processed {num_frames} frames in {elapsed:.2f}s ({fps:.1f} FPS)."
    )

    # Health diagnostics summary
    if monitor:
        final_findings = monitor.evaluate()
        print("\n" + "-" * 70)
        print(f"  SYSTEM HEALTH REPORT ({len(final_findings)} active finding(s))")
        print("-" * 70)
        if not final_findings:
            print("  [OK] No health findings. All signals within normal bounds.")
        for f in final_findings:
            print(f"  [{f.severity.value:8s}] {f.signal}")
            print(f"           {f.message[:100]}...")
            print(f"           -> {f.recommended_action[:90]}")
            print(f"           (Observed frame {f.first_observed_frame} - {f.last_observed_frame})")
            print()

        if degrade and not final_findings:
            print("  [WARN] Degraded mode was enabled but no health findings fired.")
            print("         Check threshold configuration in config/thresholds.yaml.")
        elif degrade and final_findings:
            print(f"  [OK] Degraded scenario verified: {len(final_findings)} finding(s) fired as expected.")

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mock ADB Pipeline")
    parser.add_argument(
        "--frames", type=int, default=30, help="Number of frames to process"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress frame logging"
    )
    parser.add_argument(
        "--source", default=None,
        help="Webcam index (e.g. 0) or path to a video file. If omitted, uses mock synthetic frames."
    )
    parser.add_argument(
        "--degrade", action="store_true",
        help="Inject artificial signal degradation to exercise health diagnostic checks."
    )
    args = parser.parse_args()

    run_pipeline(
        num_frames=args.frames,
        verbose=not args.quiet,
        source=args.source,
        degrade=args.degrade,
    )
