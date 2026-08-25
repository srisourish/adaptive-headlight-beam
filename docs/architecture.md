# System Architecture Specification

## Overview

The Smart Adaptive Headlight (ADB) system follows a real-time, modular, pipeline-oriented architecture designed to execute on edge hardware (NVIDIA Jetson) at >= 30 FPS.

```
       +--------------------+
       |  Camera / Video    |
       +---------+----------+
                 |
                 v
   +-------------+--------------+
   |   Perception Engine        |
   | - Object Detector (YOLO)   |
   | - Depth Estimator          |
   | - Tracker (Kalman/Flow)    |
   | - Lane & Weather Detect    |
   +-------------+--------------+
                 |
                 v
   +-------------+--------------+
   |      Fusion Engine         |
   | - Curvature Estimator      |
   | - Glare Risk Model (ML/Heu)|
   | - Accident Risk Model      |
   | - Explainability Engine    |
   +-------------+--------------+
                 |
                 v
   +-------------+--------------+
   |     Decision Engine        |
   | - Zone Mapper (Gamma)      |
   | - Asymmetric State Machine |
   | - Temporal Smoother        |
   | - Override Handler         |
   +-------------+--------------+
                 |
                 +-------------------+
                 |                   |
                 v                   v
       +---------+----------+  +-----+------------+
       |   Actuation        |  |  Dashboard WS    |
       | (Serial Bridge/LED)|  |   Backend/UI     |
       +--------------------+  +------------------+
```

## Dataflow Details

1. **Perception Engine**:
   - `CameraCapture` acquires 1080p@30fps frames (or synthetic mock stream).
   - `Detector` identifies bounding boxes for `car`, `truck`, `bus`, `motorcycle`, `person`, `bicycle`.
   - `DepthEstimator` calculates physical distances ($Z_{meters}$) using bounding box size and monocular cue regression.
   - `Tracker` assigns continuous IDs and computes velocity vectors ($v_x, v_y$).
   - `LaneDetector` extracts ego-lane boundaries and heading offset.
   - `WeatherClassifier` assesses atmospheric distortion factor.

2. **Sensor Fusion Engine**:
   - Computes dynamic features: relative velocity, vertical elevation angle, road curvature radius ($R$).
   - Calculates per-object Glare Risk score $R_{glare} \in [0, 100]$ using heuristic or XGBoost ML backend.
   - Summarizes global safety metrics and risk attributions via `ExplainabilityEngine`.

3. **Decision Engine**:
   - `ZoneMapper` projects object spatial azimuth angles $[-\theta, +\theta]$ into $N=8$ discrete matrix headlight zones.
   - Calculates target zone power: $B = B_{max} \times (1 - (R / 100)^\gamma)$.
   - `BeamStateMachine` updates global high/medium/low/matrix mode using asymmetric hysteresis (dim within 150ms, restore after 1.5s).
   - `SignalSmoother` applies exponential moving average to prevent LED flicker.

4. **Actuation Engine**:
   - `SerialBridge` formats binary frame packet: `[0xAA, 0x55, ZONE_0_PWM, ..., ZONE_N_PWM, CHECKSUM]`.
   - Transmits to Arduino MCU at 115200 baud.
