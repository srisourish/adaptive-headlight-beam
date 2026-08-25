"""
video_processor.py — Frame-sampling loop for uploaded video clips.

Reads a video from a temp-file path, samples every Nth frame,
runs the full ADB pipeline on each, and returns a list of
annotated frames + per-frame metrics for charting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from .pipeline_runner import FrameResult, PipelineRunner
from .overlay import compose_annotated_frame


@dataclass
class VideoProcessResult:
    """All output produced by processing a video clip."""
    annotated_frames: list[np.ndarray]         # Annotated BGR frames (sampled)
    frame_indices: list[int]                    # Original frame numbers that were sampled
    zone_risks_over_time: list[list[float]]     # shape: [frame_idx][zone_idx]
    beam_modes_over_time: list[str]             # BeamMode.value per sampled frame
    detection_counts: list[int]                 # Number of tracked objects per frame
    processing_times_ms: list[float]            # Pipeline latency per frame
    n_zones: int
    total_frames: int                           # Total frames in original video
    sampled_count: int                          # How many frames were processed


_BEAM_MODE_ORDER = {
    "HIGH_BEAM": 3,
    "MEDIUM_BEAM": 2,
    "MATRIX_PARTIAL": 1,
    "LOW_BEAM": 0,
}


def process_video(
    video_path: str,
    runner: PipelineRunner,
    sample_every_n: int = 5,
    max_frames: int = 300,
    progress_callback: Callable[[float], None] | None = None,
) -> VideoProcessResult:
    """
    Process a video file with the ADB pipeline, sampling every N-th frame.

    Args:
        video_path: Absolute path to the video file (mp4/mov etc.).
        runner: An already-initialised :class:`PipelineRunner`.
        sample_every_n: Process 1 in every N frames (1 = every frame).
        max_frames: Hard cap on number of frames to process (avoids UI freeze).
        progress_callback: Optional callable(fraction: float) for progress reporting.

    Returns:
        :class:`VideoProcessResult` with all annotated frames and metrics.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 0:
        total_frames = 0
    sample_every_n = max(1, sample_every_n)

    annotated_frames: list[np.ndarray] = []
    frame_indices: list[int] = []
    zone_risks_over_time: list[list[float]] = []
    beam_modes_over_time: list[str] = []
    detection_counts: list[int] = []
    processing_times_ms: list[float] = []

    frame_num = 0
    processed = 0

    while processed < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_num % sample_every_n == 0:
            result: FrameResult = runner.run_frame(frame)
            annotated = compose_annotated_frame(frame, result)

            annotated_frames.append(annotated)
            frame_indices.append(frame_num)
            zone_risks_over_time.append(result.zone_risks)
            beam_modes_over_time.append(result.beam_mode)
            detection_counts.append(len(result.detections))
            processing_times_ms.append(result.processing_time_ms)

            processed += 1

            if progress_callback is not None and total_frames > 0:
                progress_callback(min(1.0, frame_num / total_frames))

        frame_num += 1

    cap.release()

    if progress_callback is not None:
        progress_callback(1.0)

    actual_total = frame_num if total_frames <= 0 else total_frames

    return VideoProcessResult(
        annotated_frames=annotated_frames,
        frame_indices=frame_indices,
        zone_risks_over_time=zone_risks_over_time,
        beam_modes_over_time=beam_modes_over_time,
        detection_counts=detection_counts,
        processing_times_ms=processing_times_ms,
        n_zones=runner.zone_count,
        total_frames=actual_total,
        sampled_count=processed,
    )


def beam_mode_to_int(mode: str) -> int:
    """Convert beam mode string to a numeric level for plotting."""
    return _BEAM_MODE_ORDER.get(mode, 0)
