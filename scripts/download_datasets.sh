#!/usr/bin/env bash
# Dataset downloader script for smart-adaptive-headlight

set -e

DATA_DIR="data/datasets"
mkdir -p "$DATA_DIR"

echo "======================================================="
echo " Downloading Nighttime & Driving Datasets for ADB"
echo "======================================================="
echo " Target directory: $DATA_DIR"

# Download sample video stream for offline mock testing
SAMPLE_VIDEO="$DATA_DIR/night_drive_sample.mp4"

if [ ! -f "$SAMPLE_VIDEO" ]; then
    echo "Creating dummy synthetic test clip info..."
    echo "Sample night drive video download placeholder." > "$SAMPLE_VIDEO.txt"
fi

echo "Dataset downloader setup complete."
