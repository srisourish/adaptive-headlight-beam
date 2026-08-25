"""
test_webcam_and_video.py — Integration tests for webcam recording and video processing pipeline.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from dashboard.streamlit_utils.pipeline_runner import PipelineRunner
from dashboard.streamlit_utils.video_processor import process_video, VideoProcessResult
from dashboard.streamlit_utils.webcam_recorder import is_webrtc_available


def test_process_video_clip():
    """Test process_video processes a multi-frame video clip and generates time series results."""
    # Create a temporary synthetic mp4 clip with 15 frames
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_path, fourcc, 10.0, (640, 480))
    for i in range(15):
        # Create dummy frame
        img = np.zeros((480, 640, 3), dtype=np.uint8) + (i * 10)
        out.write(img)
    out.release()

    runner = PipelineRunner(mock=True, ego_speed_kmh=60.0)
    result: VideoProcessResult = process_video(
        video_path=tmp_path,
        runner=runner,
        sample_every_n=2,
        max_frames=100,
    )

    # Clean up file
    Path(tmp_path).unlink(missing_ok=True)

    assert isinstance(result, VideoProcessResult)
    assert result.sampled_count > 1
    assert len(result.zone_risks_over_time) == result.sampled_count
    assert len(result.beam_modes_over_time) == result.sampled_count
    assert len(result.annotated_frames) == result.sampled_count


def test_webrtc_availability_check():
    """Test is_webrtc_available returns a boolean without throwing."""
    res = is_webrtc_available()
    assert isinstance(res, bool)
