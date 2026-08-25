"""
dashboard/streamlit_utils/health_panel.py — System Health & Predictive Diagnostics Panel.

Renders a self-contained Streamlit UI section for the system-health module.
This panel is *intentionally separate* from the glare-risk explainability panel
(fusion/explainability.py) and should appear as a visually distinct section in
the dashboard layout.

Panel contents
--------------
1. Top-level Status Badge  : HEALTHY / WARNING / CRITICAL (worst active severity)
2. Findings List           : Scrollable, severity-colour-coded, with
                             "ongoing since frame X" for persistent conditions
3. Sparkline Charts        : 2-3 rolling-metric trend charts (confidence, latency,
                             dropout rate) using st.line_chart

Usage
-----
    from dashboard.streamlit_utils.health_panel import render_health_panel
    render_health_panel(result)   # result: FrameResult from pipeline_runner
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from diagnostics.severity import HealthFinding, HealthSeverity, worst_severity
    _HAS_DIAGNOSTICS = True
except ImportError:
    _HAS_DIAGNOSTICS = False


# ---------------------------------------------------------------------------
# Colour/icon maps
# ---------------------------------------------------------------------------
_SEVERITY_ICON = {
    "INFO":     "ℹ️",
    "WARNING":  "⚠️",
    "CRITICAL": "🔴",
}
_SEVERITY_COLOR = {
    "INFO":     "#58a6ff",   # blue
    "WARNING":  "#e3b341",   # amber
    "CRITICAL": "#f85149",   # red
}
_STATUS_CONFIG = {
    "HEALTHY":  {"icon": "🟢", "color": "#3fb950", "label": "HEALTHY"},
    "WARNING":  {"icon": "🟡", "color": "#e3b341", "label": "WARNING"},
    "CRITICAL": {"icon": "🔴", "color": "#f85149", "label": "CRITICAL"},
}


def _overall_status(findings: list[Any]) -> str:
    """Determine overall status string from active findings list."""
    if not findings:
        return "HEALTHY"
    severities = [f.severity.value if hasattr(f, "severity") else f.get("severity", "INFO")
                  for f in findings]
    if "CRITICAL" in severities:
        return "CRITICAL"
    if "WARNING" in severities:
        return "WARNING"
    return "HEALTHY"


def _finding_severity_value(f: Any) -> str:
    if hasattr(f, "severity"):
        return f.severity.value
    return f.get("severity", "INFO")


def _finding_attr(f: Any, attr: str, default: Any = "") -> Any:
    if hasattr(f, attr):
        return getattr(f, attr)
    return f.get(attr, default) if isinstance(f, dict) else default


def render_health_panel(
    findings: list[Any],
    histories: dict[str, list[float]],
    frame_idx: int = 0,
) -> None:
    """
    Render the System Health & Predictive Diagnostics panel.

    Args:
        findings : List of HealthFinding objects (from HealthMonitor.evaluate()).
        histories: Dict of metric_name → list of recent float values (for sparklines).
        frame_idx: Current frame counter (for display).
    """
    if not _HAS_DIAGNOSTICS:
        st.warning("⚠️ diagnostics module not found. Health panel disabled.")
        return

    # ------------------------------------------------------------------
    # Panel header — visually distinct from the glare-risk section above
    # ------------------------------------------------------------------
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0d1117 0%, #1a1f2e 100%);
        border: 1px solid #30363d;
        border-top: 3px solid #bc8cff;
        border-radius: 12px;
        padding: 18px 22px 12px 22px;
        margin-bottom: 0px;
    ">
        <div style="font-size: 0.80rem; font-weight:700; letter-spacing:0.12em;
                    text-transform:uppercase; color:#8b949e; margin-bottom:6px;">
            🔧 System Health &amp; Predictive Diagnostics
        </div>
        <div style="font-size: 0.72rem; color: #6e7681;">
            Monitors internal pipeline signals for early-warning patterns.
            Separate from glare-risk explainability.
        </div>
    </div>
    """, unsafe_allow_html=True)

    status = _overall_status(findings)
    cfg = _STATUS_CONFIG[status]

    # ------------------------------------------------------------------
    # Status badge
    # ------------------------------------------------------------------
    st.markdown(f"""
    <div style="
        background: {'#0d1f0d' if status == 'HEALTHY' else '#1f1a0d' if status == 'WARNING' else '#1f0d0d'};
        border: 2px solid {cfg['color']};
        border-radius: 10px;
        padding: 12px 20px;
        text-align: center;
        margin: 10px 0;
    ">
        <span style="font-size:1.5rem;">{cfg['icon']}</span>
        <span style="
            font-size: 1.1rem;
            font-weight: 700;
            color: {cfg['color']};
            letter-spacing: 0.08em;
            margin-left: 10px;
        ">{cfg['label']}</span>
        <div style="font-size:0.72rem; color:#6e7681; margin-top:4px;">
            {len(findings)} active finding(s) · Frame {frame_idx}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Findings list
    # ------------------------------------------------------------------
    if not findings:
        st.markdown(
            '<div style="color:#3fb950; font-size:0.85rem; padding:8px 4px;">'
            '✅ No active health findings. All monitored signals are within normal bounds.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:0.80rem; font-weight:600; color:#8b949e; '
            'margin-bottom:6px; text-transform:uppercase; letter-spacing:0.08em;">'
            'Active Findings</div>',
            unsafe_allow_html=True,
        )
        for f in sorted(findings, key=lambda x: _finding_severity_value(x), reverse=True):
            sev = _finding_severity_value(f)
            color = _SEVERITY_COLOR.get(sev, "#58a6ff")
            icon = _SEVERITY_ICON.get(sev, "ℹ️")
            signal = _finding_attr(f, "signal", "unknown")
            message = _finding_attr(f, "message", "")
            action = _finding_attr(f, "recommended_action", "")
            first = int(_finding_attr(f, "first_observed_frame", 0))
            last = int(_finding_attr(f, "last_observed_frame", 0))
            dur = last - first
            metric_name = _finding_attr(f, "metric_name", "")
            metric_val = _finding_attr(f, "metric_value", 0.0)

            # Duration label
            if dur > 1:
                duration_label = f"Ongoing since frame {first} ({dur} frames)"
            else:
                duration_label = f"First observed at frame {first}"

            st.markdown(f"""
            <div style="
                background: #161b22;
                border-left: 4px solid {color};
                border-radius: 0 8px 8px 0;
                padding: 12px 16px;
                margin-bottom: 10px;
            ">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <span style="font-weight:700; color:{color}; font-size:0.82rem;">
                        {icon} [{sev}] {signal.replace('_', ' ').title()}
                    </span>
                    <span style="font-size:0.70rem; color:#6e7681; font-family:'JetBrains Mono', monospace;">
                        {metric_name}: {metric_val:.3f}
                    </span>
                </div>
                <div style="font-size:0.80rem; color:#c9d1d9; margin-bottom:6px; line-height:1.45;">
                    {message}
                </div>
                <div style="font-size:0.75rem; background:#0d1117; border-radius:6px;
                            padding:6px 10px; color:#8b949e; margin-bottom:4px;">
                    💡 <b style="color:#e3b341;">Recommended:</b>&nbsp;{action}
                </div>
                <div style="font-size:0.68rem; color:#6e7681; font-family:'JetBrains Mono', monospace;">
                    {duration_label}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Sparkline charts — 3 most relevant rolling metrics
    # ------------------------------------------------------------------
    st.markdown(
        '<div style="font-size:0.80rem; font-weight:600; color:#8b949e; '
        'margin: 12px 0 6px; text-transform:uppercase; letter-spacing:0.08em;">'
        'Signal Trends</div>',
        unsafe_allow_html=True,
    )

    sparkline_metrics = [
        ("detection_conf",    "Detection Confidence",     "confidence"),
        ("frame_latency_ms",  "Frame Latency (ms)",       "ms"),
        ("detection_zero",    "Dropout Rate",             "fraction"),
    ]

    cols = st.columns(3)
    for col, (key, label, unit) in zip(cols, sparkline_metrics):
        data = histories.get(key, [])
        with col:
            st.markdown(
                f'<div style="font-size:0.72rem; color:#8b949e; text-align:center; '
                f'margin-bottom:2px;">{label}</div>',
                unsafe_allow_html=True,
            )
            if len(data) > 1:
                # Use line_chart via dict for Streamlit ≥ 1.14
                import pandas as pd
                df = pd.DataFrame({label: list(data)})
                st.line_chart(df, height=80, use_container_width=True)
                latest = data[-1]
                st.markdown(
                    f'<div style="font-size:0.68rem; color:#6e7681; text-align:center;">'
                    f'Latest: <b style="color:#e6edf3;">{latest:.3f}</b> {unit}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="font-size:0.72rem; color:#6e7681; text-align:center; '
                    'padding:20px 0;">Collecting data…</div>',
                    unsafe_allow_html=True,
                )
