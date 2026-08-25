"""
Streamlit utility modules for smart-adaptive-headlight dashboard.

Sub-modules:
    pipeline_runner — wraps perception→fusion→decision for one frame
    overlay         — draws bboxes, zone bars, mode badge onto frames
    video_processor — frame-sampling loop for uploaded video clips
"""
from __future__ import annotations
