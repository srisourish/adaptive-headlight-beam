"""
OpenCV camera calibration script using chessboard pattern.
Estimates camera intrinsic matrix K and distortion coefficients D.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def calibrate(
    image_dir: str = "data/calib_images",
    pattern_size: tuple[int, int] = (9, 6),
    square_size: float = 0.025,
    output_yaml: str = "config/camera_calib.yaml",
) -> None:
    """Run chessboard camera calibration."""
    print(f"Starting camera calibration from images in '{image_dir}'...")

    images = list(Path(image_dir).glob("*.jpg")) + list(
        Path(image_dir).glob("*.png")
    )

    if not images:
        print(
            f"No calibration images found in '{image_dir}'. Generating baseline camera config."
        )
        # Create default camera calibration dictionary
        calib_data = {
            "camera_matrix": {
                "fx": 800.0,
                "fy": 800.0,
                "cx": 640.0,
                "cy": 360.0,
            },
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
            "frame_width": 1280,
            "frame_height": 720,
            "fov_horizontal_deg": 80.0,
        }
    else:
        # Prepare 3D object points (0,0,0), (1,0,0), (2,0,0) ...
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0 : pattern_size[0], 0 : pattern_size[1]
        ].T.reshape(-1, 2)
        objp *= square_size

        objpoints = []
        imgpoints = []
        gray = None

        for img_path in images:
            img = cv2.imread(str(img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        if not objpoints or gray is None:
            print("Failed to detect chessboard corners in any provided image.")
            return

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        calib_data = {
            "camera_matrix": {
                "fx": float(mtx[0, 0]),
                "fy": float(mtx[1, 1]),
                "cx": float(mtx[0, 2]),
                "cy": float(mtx[1, 2]),
            },
            "distortion_coefficients": dist.ravel().tolist(),
            "frame_width": gray.shape[1],
            "frame_height": gray.shape[0],
            "fov_horizontal_deg": 80.0,
        }

    out_file = Path(output_yaml)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        yaml.dump(calib_data, f)

    print(f"Saved camera calibration config to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera Calibration")
    parser.add_argument("--image-dir", default="data/calib_images")
    parser.add_argument("--output", default="config/camera_calib.yaml")
    args = parser.parse_args()

    calibrate(image_dir=args.image_dir, output_yaml=args.output)
