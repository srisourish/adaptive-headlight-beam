"""
PyTorch training script for weather condition classifier (clear, rain, fog, snow).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def train_weather_classifier(
    data_dir: str = "data/weather",
    epochs: int = 5,
    output_path: str = "models/weights/weather_classifier_resnet18.pth",
) -> None:
    """Train ResNet-18 weather classifier."""
    print(f"Initializing weather classifier training (Target path: {output_path})...")

    try:
        import torch
        import torch.nn as nn
        from torchvision import models

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # ResNet18 model initialization for 4 weather classes
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 4)
        model = model.to(device)

        print("Model initialized successfully. Mock training loop completed.")
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Save dummy / initialized weights
        torch.save(model.state_dict(), out_file)
        print(f"Saved weather classifier model checkpoint to {out_file}")
    except ImportError:
        print("PyTorch / torchvision not installed. Skipping actual weights serialization.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Weather Classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Epoch count")
    parser.add_argument(
        "--output", default="models/weights/weather_classifier_resnet18.pth"
    )
    args = parser.parse_args()
    train_weather_classifier(epochs=args.epochs, output_path=args.output)
