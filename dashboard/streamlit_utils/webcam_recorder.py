"""
webcam_recorder.py — Webcam Video Capture for Smart Adaptive Headlight (ADB).

Implements Option B (Record-then-Analyze via browser HTML5 MediaRecorder API)
as the primary, robust webcam path, with optional Option A (streamlit-webrtc
live streaming) if `streamlit-webrtc` is installed.
"""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from .pipeline_runner import PipelineRunner
from .overlay import compose_annotated_frame

if TYPE_CHECKING:
    pass

# Declare custom component path
_COMPONENT_DIR = Path(__file__).parent / "webcam_component"
_webcam_component = components.declare_component(
    "webcam_recorder",
    path=str(_COMPONENT_DIR),
)


def render_webcam_recorder(key: str = "webcam_recorder_comp") -> str | None:
    """
    Renders the Option B browser webcam video recorder.

    Returns:
        Path to temporary video clip file (.webm/.mp4) if user completed recording,
        or `None` if no clip is ready or camera is uninitialized/error.
    """
    component_value = _webcam_component(key=key, default=None)

    if component_value is None:
        return None

    if isinstance(component_value, dict):
        if component_value.get("error"):
            st.warning(
                f"⚠️ {component_value['error']}"
            )
            return None

        b64_bytes = component_value.get("bytes")
        mime_type = component_value.get("mimeType", "video/webm")

        if b64_bytes:
            video_bytes = base64.b64decode(b64_bytes)
            ext = ".mp4" if "mp4" in mime_type.lower() else ".webm"

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(video_bytes)
                return tmp.name

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Option A — WebRTC Live Streamer (optional fallback / mode)
# ──────────────────────────────────────────────────────────────────────────────

def is_webrtc_available() -> bool:
    """Check if streamlit-webrtc package is available in the Python environment."""
    try:
        import streamlit_webrtc  # noqa: F401
        return True
    except ImportError:
        return False


def render_webrtc_live_streamer(runner: PipelineRunner, frame_skip: int = 3):
    """
    Renders Option A live WebRTC video streamer with real-time frame processing.

    Falls back gracefully if streamlit-webrtc is unavailable or encounters an error.
    """
    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
        import av

        class ADBVideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.runner = runner
                self.frame_count = 0
                self.last_result = None

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                self.frame_count += 1

                # Process every Nth frame for performance
                if self.frame_count % max(1, frame_skip) == 0 or self.last_result is None:
                    self.last_result = self.runner.run_frame(img)

                if self.last_result is not None:
                    annotated = compose_annotated_frame(img, self.last_result)
                else:
                    annotated = img

                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        webrtc_streamer(
            key="adb-live-webrtc",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=ADBVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    except Exception as e:
        st.error(
            "⚠️ Live webcam capture unavailable in this environment — "
            f"please use the video upload option instead. (WebRTC Error: {e})"
        )
