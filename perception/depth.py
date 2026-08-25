"""
Monocular depth estimator for smart-adaptive-headlight.

Wraps MiDaS (torch.hub) for zero-shot relative depth estimation.
A calibrated scale factor converts relative depth to approximate metres.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config

try:
    from diagnostics.health_monitor import get_monitor as _get_monitor
    _HAS_DIAGNOSTICS = True
except ImportError:
    _HAS_DIAGNOSTICS = False

# Clip bounds used to detect anomalous depth outputs (must match depth.min/max_depth_m)
_DEPTH_CLIP_LOW_M = 0.1
_DEPTH_CLIP_HIGH_M = 200.0


class DepthEstimator:
    """Monocular depth inference with mock fallback."""

    def __init__(self, model_type: str = "MiDaS_small", mock: bool = False) -> None:
        self._calib = get_config("camera_calib")
        self._scale = float(self._calib.get("depth_scale_factor", 30.0))
        self._mock = mock
        self._model = None
        self._transform = None
        self._device = "cpu"

        if not mock:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
                self._model.to(self._device).eval()
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                if model_type in ("DPT_Large", "DPT_Hybrid"):
                    self._transform = midas_transforms.dpt_transform
                else:
                    self._transform = midas_transforms.small_transform
            except Exception as exc:
                print(f"[DepthEstimator] Could not load MiDaS ({exc}); falling back to mock.")
                self._mock = True

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Compute relative depth map from a BGR frame. Returns float32 (H,W)."""
        if self._mock:
            return self._mock_depth(frame)
        import torch
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self._transform(img_rgb).to(self._device)
        with torch.no_grad():
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=frame.shape[:2],
                mode="bicubic", align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy().astype(np.float32)
        depth_map = depth_map.max() - depth_map + 1e-6
        return depth_map

    # Alias for method compatibility
    estimate = infer

    def sample_depth(self, depth_map: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        """Sample median depth in bbox and convert to metres."""
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 200.0
        patch = depth_map[y1:y2, x1:x2]
        median_relative = float(np.median(patch))
        distance_m = median_relative * self._scale / max(depth_map.max(), 1e-6)
        distance_m = float(np.clip(distance_m, _DEPTH_CLIP_LOW_M, _DEPTH_CLIP_HIGH_M))

        # --- Health diagnostics instrumentation ---
        if _HAS_DIAGNOSTICS:
            at_boundary = (
                distance_m <= _DEPTH_CLIP_LOW_M * 1.01
                or distance_m >= _DEPTH_CLIP_HIGH_M * 0.99
            )
            _get_monitor().record("depth", {"clip_fraction": 1.0 if at_boundary else 0.0})

        return distance_m

    def _mock_depth(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        depth = np.linspace(1.0, 0.1, h, dtype=np.float32)
        depth = np.tile(depth[:, None], (1, w))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        depth = depth * 0.8 + gray * 0.2
        depth += np.random.normal(0, 0.02, depth.shape).astype(np.float32)
        return np.clip(depth, 0.01, 1.0)


if __name__ == "__main__":
    from perception.camera_capture import Camera
    cam = Camera(mock=True)
    depth_est = DepthEstimator(mock=True)
    print("[Depth] Running mock demo. Press 'q' to quit.")
    for frame in cam.stream():
        dmap = depth_est.infer(frame)
        vis = ((dmap - dmap.min()) / (dmap.max() - dmap.min() + 1e-6) * 255).astype(np.uint8)
        vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        cv2.imshow("Depth", vis_color)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()
