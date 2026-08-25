"""
Camera capture module for smart-adaptive-headlight.

Wraps OpenCV VideoCapture with support for:
  - USB webcam (integer index)
  - Video file (path string)
  - CSI / GStreamer pipeline (string starting with 'nvarguscamerasrc' etc.)

Frames are undistorted using intrinsics from config/camera_calib.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

# Allow running as a module from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_config


class Camera:
    """Provides undistorted frames from a configurable video source."""

    def __init__(
        self,
        source: int | str | None = None,
        mock: bool = False,
        mock_size: tuple[int, int] = (1280, 720),
    ) -> None:
        """
        Args:
            source: Webcam index, video path, or GStreamer pipeline.
                    If *None*, reads from ``camera_calib.yaml`` ``source`` key.
            mock: If True, yield synthetic gradient frames (no real camera).
            mock_size: (width, height) for mock frames.
        """
        self._calib = get_config("camera_calib")
        self._mock = mock
        self._mock_size = mock_size
        self._frame_idx = 0

        if mock:
            self._cap = None
        else:
            if source is None:
                source = self._calib.get("source", 0)
            # GStreamer pipeline detection
            if isinstance(source, str) and not Path(source).exists():
                self._cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
            else:
                self._cap = cv2.VideoCapture(source)

            if not self._cap.isOpened():
                print(
                    f"[Camera] WARNING: Could not open source '{source}', "
                    "falling back to mock mode."
                )
                self._mock = True
                self._cap = None

        # Build undistortion maps
        intr = self._calib.get("intrinsic_matrix", {})
        fx = intr.get("fx", 800.0)
        fy = intr.get("fy", 800.0)
        cx = intr.get("cx", 640.0)
        cy = intr.get("cy", 360.0)
        self._camera_matrix = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
        )
        self._dist_coeffs = np.array(
            self._calib.get("distortion_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64
        )
        w = int(self._calib.get("frame_width", mock_size[0]))
        h = int(self._calib.get("frame_height", mock_size[1]))
        new_cam_mtx, _roi = cv2.getOptimalNewCameraMatrix(
            self._camera_matrix, self._dist_coeffs, (w, h), 1, (w, h)
        )
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self._camera_matrix,
            self._dist_coeffs,
            None,
            new_cam_mtx,
            (w, h),
            cv2.CV_16SC2,
        )

    # ----- public API -----

    def read(self) -> tuple[bool, np.ndarray]:
        """Read a single undistorted frame.

        Returns:
            (success, frame) — same contract as ``cv2.VideoCapture.read()``.
        """
        if self._mock:
            return True, self._generate_mock_frame()

        ret, frame = self._cap.read()  # type: ignore[union-attr]
        if not ret:
            return False, np.empty(0)
        frame = cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)
        return True, frame

    def stream(self) -> Generator[np.ndarray, None, None]:
        """Yield undistorted frames until the source is exhausted."""
        while True:
            ok, frame = self.read()
            if not ok:
                break
            yield frame

    def release(self) -> None:
        """Release the underlying capture object."""
        if self._cap is not None:
            self._cap.release()

    @property
    def is_mock(self) -> bool:
        return self._mock

    # ----- internals -----

    def _generate_mock_frame(self) -> np.ndarray:
        """Create a synthetic frame with moving rectangles simulating traffic."""
        w, h = self._mock_size
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Dark road
        frame[h // 2 :, :] = (40, 40, 40)
        # Sky gradient
        for y in range(h // 2):
            val = int(20 + 30 * (y / (h // 2)))
            frame[y, :] = (val, val, val + 10)

        # Simulated headlights (oncoming vehicle)
        cx = int((w // 4) + (self._frame_idx * 3) % (w // 2))
        cy = h // 2 + 50
        cv2.circle(frame, (cx, cy), 18, (200, 220, 255), -1)
        cv2.circle(frame, (cx + 40, cy), 18, (200, 220, 255), -1)

        # Pedestrian rectangle
        px = int(w * 0.7 + 30 * np.sin(self._frame_idx * 0.05))
        py = h // 2 + 30
        cv2.rectangle(frame, (px, py), (px + 25, py + 70), (0, 180, 0), -1)

        # Lane lines
        cv2.line(frame, (w // 3, h // 2), (0, h), (0, 200, 200), 2)
        cv2.line(frame, (2 * w // 3, h // 2), (w, h), (0, 200, 200), 2)
        cv2.line(
            frame,
            (w // 2, h // 2),
            (w // 2 - 50, h),
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        self._frame_idx += 1
        return frame


# Alias for backward compatibility and test consistency
CameraCapture = Camera


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Camera capture demo")
    parser.add_argument("--source", default=None, help="Video source")
    parser.add_argument("--mock", action="store_true", help="Use synthetic frames")
    parser.add_argument(
        "--demo", action="store_true", help="Alias for --mock (quickstart)"
    )
    args = parser.parse_args()

    cam = Camera(source=args.source, mock=args.mock or args.demo)
    print(f"[Camera] Running in {'MOCK' if cam.is_mock else 'LIVE'} mode. Press 'q' to quit.")

    for frame in cam.stream():
        cv2.imshow("Camera Capture", frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
