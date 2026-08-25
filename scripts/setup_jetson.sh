#!/usr/bin/env bash
# Environment setup script for NVIDIA Jetson Orin/Xavier running JetPack

set -e

echo "======================================================="
echo " Setting up Smart Adaptive Headlight on NVIDIA Jetson"
echo "======================================================="

# Update APT packages
sudo apt-get update && sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libopenblas-dev \
    libopencv-dev \
    v4l-utils \
    git

# Upgrade pip
python3 -m pip install --upgrade pip

# Install PyTorch for Jetson (Wheel target depending on JetPack version)
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Configure serial permissions for Arduino connection
sudo usermod -a -G dialout $USER

echo "Setup complete! Please log out and back in for serial permissions to take effect."
