"""
Weather classifier for smart-adaptive-headlight.

Provides two paths:
  1. Rule-based fallback (Laplacian variance + contrast) — works out of box.
  2. MobileNetV3-small CNN classifier (5 classes: clear/rain/fog/dust/snow)
     loaded from models/weights/ after training.
"""

from __future__ import annotations

import sys
from enum import IntEnum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from diagnostics.health_monitor import get_monitor as _get_monitor
    _HAS_DIAGNOSTICS = True
except ImportError:
    _HAS_DIAGNOSTICS = False


class WeatherClass(IntEnum):
    CLEAR = 0
    RAIN = 1
    FOG = 2
    DUST = 3
    SNOW = 4


WEATHER_NAMES = {w: w.name.lower() for w in WeatherClass}


class WeatherClassifier:
    """Weather classification with rule-based fallback and optional CNN."""

    def __init__(self, model_path: Optional[str] = None, mock: bool = False) -> None:
        self._mock = mock
        self._model = None
        self._transform = None

        if model_path and not mock:
            try:
                import torch
                import torchvision.transforms as T
                self._model = torch.jit.load(model_path, map_location="cpu")
                self._model.eval()
                self._transform = T.Compose([
                    T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
            except Exception as exc:
                print(f"[WeatherClassifier] CNN load failed ({exc}); using rule-based.")

    def classify(self, frame: np.ndarray) -> tuple[WeatherClass, float]:
        """Classify weather in a frame.

        Returns:
            (weather_class, confidence)
        """
        if self._mock:
            result = (WeatherClass.CLEAR, 0.9)
            used_fallback = True  # mock always uses "fallback" (no CNN)
        elif self._model is not None:
            result = self._cnn_classify(frame)
            used_fallback = False
        else:
            result = self._rule_based(frame)
            used_fallback = True

        # --- Health diagnostics instrumentation ---
        if _HAS_DIAGNOSTICS:
            _get_monitor().record("weather", {"used_fallback": used_fallback})

        return result

    def _cnn_classify(self, frame: np.ndarray) -> tuple[WeatherClass, float]:
        import torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            cls_id = int(probs.argmax())
            conf = float(probs[cls_id])
        return WeatherClass(cls_id), conf

    def _rule_based(self, frame: np.ndarray) -> tuple[WeatherClass, float]:
        """Heuristic weather classification using image statistics."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Laplacian variance → sharpness (low = fog/rain)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Overall brightness and contrast
        mean_val = float(gray.mean())
        std_val = float(gray.std())
        # Saturation from HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat_mean = float(hsv[:, :, 1].mean())

        # Decision tree (simplified heuristic)
        if lap_var < 50 and mean_val > 150:
            return WeatherClass.FOG, 0.7
        if lap_var < 80 and std_val < 40:
            return WeatherClass.FOG, 0.6
        if sat_mean < 30 and mean_val > 180:
            return WeatherClass.SNOW, 0.5
        if sat_mean < 40 and lap_var < 100:
            return WeatherClass.RAIN, 0.5
        if mean_val > 160 and sat_mean > 60:
            return WeatherClass.DUST, 0.4
        return WeatherClass.CLEAR, 0.8


if __name__ == "__main__":
    from perception.camera_capture import Camera
    cam = Camera(mock=True)
    wc = WeatherClassifier(mock=False)  # Use rule-based
    print("[WeatherClassifier] Running rule-based demo. Press 'q' to quit.")
    for frame in cam.stream():
        cls, conf = wc.classify(frame)
        label = f"Weather: {WEATHER_NAMES[cls]} ({conf:.2f})"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        cv2.imshow("Weather", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()
