"""
overlay.py — Draw visual annotations onto BGR frames.

All drawing functions are pure OpenCV/NumPy; no Streamlit imports here
so this module stays testable in isolation.
"""
from __future__ import annotations

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (BGR)
# ──────────────────────────────────────────────────────────────────────────────

_CLS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (60, 220, 60),    # person  — green
    1: (255, 180, 0),    # bicycle — cyan-yellow
    2: (0, 180, 255),    # car     — orange
    3: (220, 100, 255),  # motorcycle — violet
    5: (0, 80, 255),     # bus     — red-orange
    7: (0, 40, 200),     # truck   — deep red
}
_DEFAULT_COLOR: tuple[int, int, int] = (200, 200, 200)

_BEAM_COLORS: dict[str, tuple[int, int, int]] = {
    "HIGH_BEAM":      (0, 220, 255),   # amber-yellow
    "MEDIUM_BEAM":    (0, 180, 50),    # green
    "LOW_BEAM":       (255, 80, 30),   # blue-ish
    "MATRIX_PARTIAL": (30, 160, 255),  # orange
}

# Risk heat: 0→green, 50→yellow, 100→red  (BGR)
def _risk_color(risk: float) -> tuple[int, int, int]:
    """Return a BGR colour interpolated from green→yellow→red."""
    t = float(np.clip(risk / 100.0, 0, 1))
    if t < 0.5:
        r = int(255 * t * 2)
        g = 255
    else:
        r = 255
        g = int(255 * (1 - (t - 0.5) * 2))
    return (0, g, r)


# ──────────────────────────────────────────────────────────────────────────────
# Public drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def draw_detections(
    frame: np.ndarray,
    detections: list,   # list[DetectionInfo]
    draw_distance: bool = True,
    mock_mode: bool = False,
) -> np.ndarray:
    """
    Draw bounding boxes, labels, confidence, and optional distance.

    Args:
        frame: BGR image to annotate (modified **in-place** on a copy).
        detections: list of :class:`~streamlit_utils.pipeline_runner.DetectionInfo`.
        draw_distance: Whether to render the estimated distance below the label.
        mock_mode: If True, adds a "(MOCK)" suffix to every label.

    Returns:
        Annotated copy of the frame.
    """
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = _CLS_COLORS.get(det.cls, _DEFAULT_COLOR)

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Risk-tinted fill (semi-transparent top bar)
        risk_col = _risk_color(det.risk_score)
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, max(0, y1 - 20)), (x2, y1), risk_col, -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

        # Label text
        suffix = " (MOCK)" if mock_mode else ""
        label = f"ID{det.track_id} {det.cls_name}{suffix} {det.conf:.2f}"
        if draw_distance:
            label += f" | {det.distance_m:.1f}m"
        cv2.putText(
            out, label,
            (x1 + 2, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Risk score badge (bottom-right of box)
        risk_label = f"Risk:{det.risk_score:.0f}%"
        cv2.putText(
            out, risk_label,
            (x1 + 2, min(out.shape[0] - 4, y2 + 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, risk_col, 1, cv2.LINE_AA,
        )

    return out


def draw_zone_overlay(
    frame: np.ndarray,
    zone_risks: list[float],
    zone_brightness: list[int],
    n_zones: int,
) -> np.ndarray:
    """
    Draw vertical zone boundary lines and bottom risk/brightness bars onto the frame.

    Args:
        frame: BGR frame.
        zone_risks: Risk score per zone [0-100].
        zone_brightness: PWM brightness per zone [0-255].
        n_zones: Total number of zones.

    Returns:
        Annotated copy.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    zone_w = w // n_zones
    bar_h = 28  # pixel height of the bottom bar strip

    for i in range(n_zones):
        x_start = i * zone_w
        x_end = min((i + 1) * zone_w, w)
        risk = zone_risks[i] if i < len(zone_risks) else 0.0
        brt = zone_brightness[i] if i < len(zone_brightness) else 0

        col = _risk_color(risk)

        # Vertical boundary line
        if i > 0:
            cv2.line(out, (x_start, h // 2), (x_start, h), (180, 180, 180), 1, cv2.LINE_AA)

        # Bottom risk bar
        overlay = out.copy()
        cv2.rectangle(overlay, (x_start, h - bar_h), (x_end, h), col, -1)
        cv2.addWeighted(overlay, 0.70, out, 0.30, 0, out)

        # Zone label text
        cv2.putText(
            out,
            f"Z{i}\n{risk:.0f}%",
            (x_start + 3, h - bar_h + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return out


def draw_beam_badge(
    frame: np.ndarray,
    beam_mode: str,
    mock_mode: bool = False,
) -> np.ndarray:
    """
    Render a prominent beam mode badge in the top-left corner of the frame.

    Args:
        frame: BGR frame.
        beam_mode: String name of the current BeamMode (e.g. 'HIGH_BEAM').
        mock_mode: If True, suffix with 'MOCK'.

    Returns:
        Frame with badge drawn.
    """
    out = frame.copy()
    col = _BEAM_COLORS.get(beam_mode, (200, 200, 200))
    suffix = " | MOCK" if mock_mode else ""
    text = f"⬡ {beam_mode.replace('_', ' ')}{suffix}"

    # Badge background
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
    pad = 8
    overlay = out.copy()
    cv2.rectangle(overlay, (8, 8), (8 + tw + pad * 2, 8 + th + pad * 2 + baseline), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
    cv2.rectangle(out, (8, 8), (8 + tw + pad * 2, 8 + th + pad * 2 + baseline), col, 2)

    cv2.putText(
        out, text,
        (8 + pad, 8 + pad + th),
        cv2.FONT_HERSHEY_DUPLEX, 0.7, col, 2, cv2.LINE_AA,
    )
    return out


def draw_mock_watermark(frame: np.ndarray) -> np.ndarray:
    """Stamp a SYNTHETIC DATA watermark diagonally across the frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    text = "SYNTHETIC / MOCK DATA"
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw once across the image diagonal
    overlay = out.copy()
    cv2.putText(
        overlay, text,
        (w // 8, h // 2),
        font, 1.2, (0, 0, 255), 2, cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, 0.20, out, 0.80, 0, out)
    return out


def compose_annotated_frame(
    frame: np.ndarray,
    result,          # FrameResult
    show_mock_watermark: bool = True,
) -> np.ndarray:
    """
    Compose full annotated frame: detections + zone overlay + beam badge.

    Args:
        frame: Original BGR frame.
        result: :class:`~streamlit_utils.pipeline_runner.FrameResult`.
        show_mock_watermark: Draw watermark if any mock component is active.

    Returns:
        Fully annotated BGR frame, ready for :func:`st.image`.
    """
    any_mock = any(result.mock_flags.values())
    out = draw_detections(frame, result.detections, mock_mode=any_mock)
    out = draw_zone_overlay(out, result.zone_risks, result.zone_brightness, result.n_zones)
    out = draw_beam_badge(out, result.beam_mode, mock_mode=any_mock)
    if any_mock and show_mock_watermark:
        out = draw_mock_watermark(out)
    return out
