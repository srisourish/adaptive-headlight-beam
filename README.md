# Smart Adaptive Headlight (ADB) System

An AI-driven, edge-accelerated Matrix LED Adaptive Driving Beam (ADB) headlight system for modern automotive safety. 
Combines computer vision (YOLOv8 + MiDaS depth estimation), optical flow tracking, weather/road curvature fusion, multi-zone glare risk assessment, and low-latency serial actuation to selectively dim Matrix LED zones without blinding oncoming or preceding drivers.

---

## 🌟 Key Features

- **Multi-Modal Perception**:
  - **Object Detection**: YOLOv8 vehicle, pedestrian, and cyclist detection with mock video support.
  - **Depth Estimation**: Monocular depth inference for real-world distance calculation.
  - **Object Tracking**: Kalman filter / optical flow tracking across temporal frames.
  - **Lane & Curvature Estimation**: Hough transform + parabolic polynomial road profile estimation.
  - **Weather Classification**: Vision-based environmental state detection (clear, rain, fog, snow).
  - **Sign Recognition**: Detects retroreflective traffic signs to mitigate high-beam self-glare.
- **Sensor Fusion & Glare Risk**:
  - Heuristic and Gradient Boosted (XGBoost/LightGBM) dual-backend glare risk models.
  - Explainability module quantifying per-feature risk attribution.
- **Decision Engine**:
  - Asymmetric hysteresis state machine protecting drivers instantly (<150ms) while restoring high-beams smoothly (1.5s).
  - Gamma-corrected N-zone spatial power mapping.
  - Telemetry override handler for manual driver inputs.
- **Actuation & Hardware**:
  - Hardware-in-the-loop Serial Bridge communicating with Arduino Matrix LED controllers via custom binary frame protocol.
  - 8-zone high-speed PWM LED dimmer.
- **Live Telemetry Dashboard**:
  - Real-time FastAPI + WebSocket backend with HTML5 Glassmorphism dashboard UI.
- **CARLA Co-Simulation**:
  - Automated scenario runner for CARLA synthetic night-driving evaluation.

---

## 📁 Repository Structure

```
smart-adaptive-headlight/
├── README.md                           # System documentation
├── LICENSE                             # MIT License
├── requirements.txt                    # Project dependencies
├── docs/                               # System specifications
│   ├── architecture.md                 # Dataflow and system pipeline
│   ├── glare_risk_model.md             # Mathematical risk formulations
│   ├── hardware_setup.md               # Jetson & Arduino setup guide
│   └── paper/                          # Academic paper outline
├── perception/                         # Computer vision pipeline
│   ├── camera_capture.py               # Video capture & synthetic stream
│   ├── detector.py                     # YOLOv8 object detector
│   ├── depth.py                        # Monocular depth estimator
│   ├── tracker.py                      # Multi-object tracker
│   ├── lane_detector.py                # Lane line detection
│   ├── weather_classifier.py           # Weather condition classifier
│   └── sign_recognizer.py              # Traffic sign recognizer
├── fusion/                             # Risk assessment & fusion
│   ├── object_classifier.py            # Target classification
│   ├── curvature_estimator.py          # Road curvature modeling
│   ├── glare_risk_model.py             # Dual-backend glare risk score
│   ├── accident_risk_model.py          # Multi-vehicle accident risk score
│   └── explainability.py               # SHAP/feature risk breakdown
├── decision/                           # Beam mode & power allocation
│   ├── beam_state_machine.py           # Hysteresis state machine
│   ├── zone_mapper.py                  # Pixel-to-zone gamma brightness
│   ├── smoothing.py                    # Temporal exponential filter
│   └── override_handler.py             # Telemetry & driver override
├── actuation/                          # Matrix LED hardware bridge
│   ├── serial_bridge.py                # Python serial bridge protocol
│   └── arduino/beam_controller.ino     # C++ Arduino matrix LED controller
├── dashboard/                          # Live web dashboard
│   ├── backend/main.py                 # FastAPI + WebSockets server
│   └── frontend/                       # HTML/CSS/JS web UI
├── sim/                                # Simulation runner
│   ├── carla_scenario_runner.py        # CARLA integration bridge
│   └── scenario_configs/               # Scenario JSON configs
├── models/                             # Training & model weights
│   ├── weights/                        # Model weights directory
│   └── training/                       # Fine-tuning & model training
├── config/                             # System YAML configuration
│   ├── zones.yaml                      # Angular zone definitions
│   ├── thresholds.yaml                 # Hysteresis and speed thresholds
│   └── camera_calib.yaml               # Intrinsics and extrinsic matrices
├── tests/                              # Unit, integration, and HIL tests
│   ├── unit/                           # Pytest unit tests
│   ├── integration/                    # Pipeline integration runner
│   └── hardware_in_loop/               # Serial HIL testing
└── scripts/                            # Installation & utility scripts
    ├── setup_jetson.sh                 # NVIDIA Jetson environment setup
    ├── calibrate_camera.py             # OpenCV camera calibration tool
    └── download_datasets.sh            # Dataset download script
```

---

## ⚡ Quick Start

### 1. Installation

```bash
git clone https://github.com/your-org/smart-adaptive-headlight.git
cd smart-adaptive-headlight
pip install -r requirements.txt
```

### 2. Run Mock Pipeline (No Hardware Required)

Execute the end-to-end perception → fusion → decision → actuation mock loop on synthetic frames:

```bash
python -m tests.integration.run_mock_pipeline --frames 30
```

### 3. Run Unit Tests

```bash
pytest tests/unit
```

### 4. Launch Live Web Dashboard

```bash
python dashboard/backend/main.py --mock --port 8000
```
Open your browser to `http://localhost:8000`.

---

## 🔧 Hardware Deployment

For Jetson Orin / Xavier deployment connected to an Arduino Matrix LED controller:

1. Flash `actuation/arduino/beam_controller.ino` using `arduino-cli` or Arduino IDE.
2. Connect Jetson USB to Arduino (`/dev/ttyACM0`).
3. Run Jetson setup script:
   ```bash
   bash scripts/setup_jetson.sh
   ```

---

## 📄 Citation & License

Distributed under the MIT License. See `LICENSE` for more information.
