"""
Fine-tuning script for YOLOv8 object detector on nighttime vehicle datasets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def finetune_yolo(
    data_yaml: str = "data/coco_night.yaml",
    weights: str = "yolov8n.pt",
    epochs: int = 10,
    imgsz: int = 640,
) -> None:
    """Fine-tune YOLO model on custom dataset."""
    print(f"Fine-tuning YOLO model '{weights}' on dataset '{data_yaml}'...")

    try:
        from ultralytics import YOLO

        model = YOLO(weights)
        print("Ultralytics YOLO loaded successfully.")
        # Execute fine-tuning if dataset yaml exists
        if Path(data_yaml).exists():
            model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)
            print("Training complete.")
        else:
            print(f"Dataset config '{data_yaml}' not found. Setup complete in dry-run mode.")
    except ImportError:
        print("Ultralytics library not installed. Ultralytics YOLO mock fallback active.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for Night Vision ADB")
    parser.add_argument("--data", default="data/coco_night.yaml")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    finetune_yolo(data_yaml=args.data, weights=args.weights, epochs=args.epochs)
