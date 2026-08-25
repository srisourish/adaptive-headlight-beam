"""
streamlit_app.py — Smart Adaptive Headlight (ADB) Demo Dashboard.

Run with:
    streamlit run dashboard/streamlit_app.py

From the project root.  No hardware, camera, or Arduino required.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Dashboard sub-modules ─────────────────────────────────────────────────────
from dashboard.streamlit_utils.pipeline_runner import PipelineRunner, FrameResult
from dashboard.streamlit_utils.overlay import compose_annotated_frame
from dashboard.streamlit_utils.video_processor import (
    process_video,
    beam_mode_to_int,
    VideoProcessResult,
)
from dashboard.streamlit_utils.webcam_recorder import (
    render_webcam_recorder,
    render_webrtc_live_streamer,
    is_webrtc_available,
)
try:
    from dashboard.streamlit_utils.health_panel import render_health_panel
    _HAS_HEALTH_PANEL = True
except Exception:
    _HAS_HEALTH_PANEL = False

# ──────────────────────────────────────────────────────────────────────────────
# Page config — must be the FIRST Streamlit call
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Adaptive Headlight — ADB Demo",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS — premium dark dashboard aesthetic
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root tokens ── */
:root {
    --bg-primary:    #0d1117;
    --bg-secondary:  #161b22;
    --bg-card:       #1c2230;
    --accent-blue:   #58a6ff;
    --accent-amber:  #e3b341;
    --accent-green:  #3fb950;
    --accent-red:    #f85149;
    --accent-purple: #bc8cff;
    --text-primary:  #e6edf3;
    --text-secondary:#8b949e;
    --border:        #30363d;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}

/* ── Main area ── */
.main .block-container {
    padding-top: 1.5rem;
    max-width: 1400px;
}

/* ── Cards ── */
.adb-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

/* ── Beam mode badge ── */
.beam-badge {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 24px;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-align: center;
    width: 100%;
    margin: 8px 0;
}
.HIGH_BEAM      { background: #1a2d1a; color: #3fb950; border: 2px solid #3fb950; }
.MEDIUM_BEAM    { background: #2a2512; color: #e3b341; border: 2px solid #e3b341; }
.LOW_BEAM       { background: #1d1229; color: #bc8cff; border: 2px solid #bc8cff; }
.MATRIX_PARTIAL { background: #1d2030; color: #58a6ff; border: 2px solid #58a6ff; }

/* ── Mock warning banner ── */
.mock-banner {
    background: #2d1e0f;
    border-left: 4px solid #e3b341;
    padding: 10px 16px;
    border-radius: 6px;
    font-size: 0.82rem;
    color: #d4a843;
    margin-bottom: 12px;
}

/* ── Explainability panel ── */
.explain-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
}
.explain-key  { color: var(--text-secondary); }
.explain-val  { color: var(--accent-blue); font-family: 'JetBrains Mono', monospace; }

/* ── Metric overrides ── */
[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 18px;
}

/* ── Section headers ── */
.section-header {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-secondary);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: var(--bg-card);
    border-radius: 8px;
    color: var(--text-secondary);
    border: 1px solid var(--border);
}
.stTabs [aria-selected="true"] {
    background: var(--bg-secondary) !important;
    color: var(--accent-blue) !important;
    border-color: var(--accent-blue) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #58a6ff, #3fb950);
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Cached model loader — runs once per (mock_flag, n_zones) combination
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️  Loading pipeline models…")
def get_pipeline(mock: bool, n_zones: int, ego_speed_kmh: float) -> PipelineRunner:
    """Return a cached :class:`PipelineRunner`."""
    return PipelineRunner(
        mock=mock,
        n_zones_override=n_zones if n_zones else None,
        ego_speed_kmh=ego_speed_kmh,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

_BEAM_ICONS = {
    "HIGH_BEAM":      "🌟",
    "MEDIUM_BEAM":    "💡",
    "LOW_BEAM":       "🔅",
    "MATRIX_PARTIAL": "🔷",
}

_BEAM_DESC = {
    "HIGH_BEAM":      "All zones clear — full long-range illumination active.",
    "MEDIUM_BEAM":    "All zones safe but speed low — moderate reach.",
    "LOW_BEAM":       "High glare risk or adverse weather — protective short-range mode.",
    "MATRIX_PARTIAL": "Selected zones dimmed — ADB matrix actively masking oncoming traffic.",
}


def build_mock_status_html(flags: dict[str, bool]) -> str:
    """Build HTML for the mock-mode status table shown in the sidebar."""
    rows = ""
    for comp, is_mock in flags.items():
        icon = "🟡 MOCK" if is_mock else "🟢 REAL"
        rows += f"<tr><td style='padding:2px 8px;color:#8b949e;'>{comp}</td><td style='padding:2px 8px;'>{icon}</td></tr>"
    return f"<table style='font-size:0.78rem;border-collapse:collapse;'>{rows}</table>"


def risk_bar_figure(zone_risks: list[float], zone_brightness: list[int]):
    """Return a Plotly figure with per-zone risk bars and brightness line."""
    try:
        import plotly.graph_objects as go
        n = len(zone_risks)
        labels = [f"Z{i}" for i in range(n)]

        # Colour per bar based on risk
        colors = []
        for r in zone_risks:
            t = r / 100.0
            if t < 0.5:
                r_val = int(255 * t * 2)
                g_val = 255
            else:
                r_val = 255
                g_val = int(255 * (1 - (t - 0.5) * 2))
            colors.append(f"rgb({r_val},{g_val},0)")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=zone_risks, name="Glare Risk (%)",
            marker_color=colors, opacity=0.85,
            yaxis="y1",
        ))
        fig.add_trace(go.Scatter(
            x=labels,
            y=[b / 255.0 * 100 for b in zone_brightness],
            name="Brightness (% max)", mode="lines+markers",
            line=dict(color="#58a6ff", width=2),
            marker=dict(size=6),
            yaxis="y1",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1c2230",
            plot_bgcolor="#1c2230",
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.05, x=0, font=dict(size=10)),
            yaxis=dict(range=[0, 105], title="", gridcolor="#30363d"),
            xaxis=dict(gridcolor="#30363d"),
            font=dict(family="Inter", color="#e6edf3", size=11),
        )
        return fig
    except ImportError:
        return None


def risk_over_time_figure(frame_indices: list[int], zone_risks_over_time: list[list[float]], n_zones: int):
    """Return Plotly line chart of per-zone risk over time."""
    try:
        import plotly.graph_objects as go
        ZONE_PALETTE = [
            "#58a6ff", "#3fb950", "#e3b341", "#f85149",
            "#bc8cff", "#79c0ff", "#56d364", "#ffab70",
        ]
        fig = go.Figure()
        zone_matrix = np.array(zone_risks_over_time)  # (T, N)
        for z in range(n_zones):
            col = ZONE_PALETTE[z % len(ZONE_PALETTE)]
            fig.add_trace(go.Scatter(
                x=frame_indices, y=zone_matrix[:, z].tolist(),
                name=f"Zone {z}", mode="lines",
                line=dict(color=col, width=1.5),
            ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1c2230", plot_bgcolor="#1c2230",
            height=240, margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(title="Frame index", gridcolor="#30363d"),
            yaxis=dict(title="Glare Risk (%)", range=[0, 105], gridcolor="#30363d"),
            legend=dict(orientation="h", y=1.08, font=dict(size=9)),
            font=dict(family="Inter", color="#e6edf3", size=11),
        )
        return fig
    except ImportError:
        return None


def beam_over_time_figure(frame_indices: list[int], beam_modes: list[str]):
    """Return Plotly step chart of beam mode level over time."""
    try:
        import plotly.graph_objects as go
        levels = [beam_mode_to_int(m) for m in beam_modes]
        mode_names = {0: "LOW_BEAM", 1: "MATRIX_PARTIAL", 2: "MEDIUM_BEAM", 3: "HIGH_BEAM"}
        colors_map = {0: "#bc8cff", 1: "#58a6ff", 2: "#e3b341", 3: "#3fb950"}

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frame_indices, y=levels,
            mode="lines", line_shape="hv",
            line=dict(color="#58a6ff", width=2),
            fill="tozeroy", fillcolor="rgba(88,166,255,0.12)",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1c2230", plot_bgcolor="#1c2230",
            height=160, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Frame index", gridcolor="#30363d"),
            yaxis=dict(
                title="Beam Level",
                tickvals=[0, 1, 2, 3],
                ticktext=["LOW", "MATRIX", "MEDIUM", "HIGH"],
                range=[-0.2, 3.5], gridcolor="#30363d",
            ),
            font=dict(family="Inter", color="#e6edf3", size=11),
        )
        return fig
    except ImportError:
        return None


def process_and_render_video(
    video_path: str,
    runner: PipelineRunner,
    sample_n: int,
    key_prefix: str = "vid",
) -> None:
    """
    Process a video file (upload or webcam clip) with process_video
    and render full interactive metrics, risk-over-time, and frame carousel.
    """
    progress_bar = st.progress(0, text="Analysing frames…")

    def _update_progress(frac: float) -> None:
        progress_bar.progress(min(frac, 1.0), text=f"Analysing frames… {frac:.0%}")

    vid_result: VideoProcessResult = process_video(
        video_path=video_path,
        runner=runner,
        sample_every_n=sample_n,
        max_frames=300,
        progress_callback=_update_progress,
    )
    progress_bar.empty()

    st.success(
        f"✅ Processed **{vid_result.sampled_count}** frames "
        f"(of {vid_result.total_frames} total, sampled every {sample_n})."
    )

    # ── Frame carousel ─────────────────────────────────────────────────
    st.markdown("### 🎞️ Sampled Frames")
    if vid_result.annotated_frames:
        frame_slider = st.slider(
            "Browse frames",
            min_value=0,
            max_value=max(0, len(vid_result.annotated_frames) - 1),
            value=0,
            key=f"{key_prefix}_frame_slider",
        )
        fr = vid_result.annotated_frames[frame_slider]
        actual_idx = vid_result.frame_indices[frame_slider]
        beam_at = vid_result.beam_modes_over_time[frame_slider]
        risk_at = vid_result.zone_risks_over_time[frame_slider]
        objs_at = vid_result.detection_counts[frame_slider]

        vcol1, vcol2 = st.columns([3, 2], gap="medium")
        with vcol1:
            st.image(
                cv2.cvtColor(fr, cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption=f"Frame #{actual_idx}  |  Mode: {beam_at}  |  Objects: {objs_at}",
            )
        with vcol2:
            icon2 = _BEAM_ICONS.get(beam_at, "💡")
            st.markdown(
                f"<div class='beam-badge {beam_at}'>{icon2} {beam_at.replace('_', ' ')}</div>",
                unsafe_allow_html=True,
            )
            st.caption(_BEAM_DESC.get(beam_at, ""))
            st.markdown("<br>", unsafe_allow_html=True)
            fig_bar2 = risk_bar_figure(risk_at, [128] * len(risk_at))
            if fig_bar2:
                st.plotly_chart(fig_bar2, use_container_width=True, config={"displayModeBar": False})

    # ── Risk over time ─────────────────────────────────────────────────
    st.markdown("### 📈 Per-Zone Glare Risk Over Time")
    fig_risk = risk_over_time_figure(
        vid_result.frame_indices,
        vid_result.zone_risks_over_time,
        vid_result.n_zones,
    )
    if fig_risk:
        st.plotly_chart(fig_risk, use_container_width=True, config={"displayModeBar": False})
    else:
        import pandas as pd
        risk_df = pd.DataFrame(
            vid_result.zone_risks_over_time,
            index=vid_result.frame_indices,
            columns=[f"Z{i}" for i in range(vid_result.n_zones)],
        )
        st.line_chart(risk_df, height=240)

    # ── Beam mode over time ────────────────────────────────────────────
    st.markdown("### 🔀 Beam Mode Over Time")
    fig_beam = beam_over_time_figure(
        vid_result.frame_indices, vid_result.beam_modes_over_time
    )
    if fig_beam:
        st.plotly_chart(fig_beam, use_container_width=True, config={"displayModeBar": False})
    else:
        import pandas as pd
        beam_df = pd.DataFrame({
            "Beam Level": [beam_mode_to_int(m) for m in vid_result.beam_modes_over_time]
        }, index=vid_result.frame_indices)
        st.line_chart(beam_df, height=160)

    # ── Summary stats ──────────────────────────────────────────────────
    with st.expander("📊 Processing Stats", expanded=False):
        avg_lat = np.mean(vid_result.processing_times_ms) if vid_result.processing_times_ms else 0
        max_lat = np.max(vid_result.processing_times_ms) if vid_result.processing_times_ms else 0
        scol1, scol2, scol3, scol4 = st.columns(4)
        scol1.metric("Sampled Frames", vid_result.sampled_count)
        scol2.metric("Total Frames", vid_result.total_frames)
        scol3.metric("Avg Latency", f"{avg_lat:.0f} ms")
        scol4.metric("Peak Latency", f"{max_lat:.0f} ms")


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────


with st.sidebar:
    st.markdown("## 💡 ADB Control Panel")
    st.markdown("---")

    use_mock = st.toggle(
        "🧪 Force Mock / Synthetic Mode",
        value=True,
        help="When ON, all perception modules produce synthetic outputs — no model weights required.",
    )

    n_zones = st.slider(
        "Beam Zones",
        min_value=4, max_value=16, value=8, step=1,
        help="Number of independently controlled LED zones (config default: 8).",
    )

    ego_speed = st.slider(
        "Ego Speed (km/h)",
        min_value=0, max_value=160, value=65,
        help="Simulated vehicle speed used by the beam state machine.",
    )

    sample_n = st.slider(
        "Video Frame Sampling (every N frames)",
        min_value=1, max_value=30, value=5,
        help="Process 1 in every N frames for video input. Higher = faster but coarser.",
    )

    st.markdown("---")
    st.markdown("#### 🔍 Pipeline Mode")

    # Instantiate pipeline and show component status
    runner = get_pipeline(mock=use_mock, n_zones=n_zones, ego_speed_kmh=float(ego_speed))
    flags = runner.mock_flags
    st.markdown(build_mock_status_html(flags), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### ℹ️ Disclaimer")
    mock_items = [comp for comp, is_mock in flags.items() if is_mock]
    if mock_items:
        st.markdown(
            f"<div class='mock-banner'>⚠️ <b>SYNTHETIC DATA</b><br>"
            f"Mocked components: <b>{', '.join(mock_items)}</b>.<br>"
            f"Outputs are representative, not real sensor measurements.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.success("✅ All modules running in real mode.")

    st.markdown(
        "<span style='font-size:0.72rem;color:#8b949e;'>"
        "SerialBridge is always mocked (no Arduino attached).</span>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Hero header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 16px 0 8px 0;'>
    <h1 style='font-size:2.1rem; font-weight:700; margin:0;
               background: linear-gradient(135deg, #58a6ff 0%, #3fb950 100%);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        ⚡ Smart Adaptive Headlight — ADB Demo
    </h1>
    <p style='color:#8b949e; font-size:0.9rem; margin-top:4px;'>
        Perception → Glare Risk Fusion → Beam Decision — browser-based, no hardware required
    </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Input tabs
# ──────────────────────────────────────────────────────────────────────────────
tab_image, tab_video, tab_webcam = st.tabs(["📷 Image Upload", "🎬 Video Upload", "📹 Webcam"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Image Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab_image:
    uploaded_img = st.file_uploader(
        "Upload a dashcam image (JPG / PNG)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_uploader",
    )

    # Default synthetic frame if nothing uploaded
    if uploaded_img is None:
        st.markdown(
            "<div class='mock-banner'>ℹ️ No image uploaded — showing a synthetic demo frame.</div>",
            unsafe_allow_html=True,
        )
        # Generate a synthetic camera frame via the mock Camera
        sys.path.insert(0, str(_ROOT))
        from perception.camera_capture import Camera
        _demo_cam = Camera(mock=True, mock_size=(1280, 720))
        _, demo_frame = _demo_cam.read()
        for _ in range(12):
            _, demo_frame = _demo_cam.read()  # advance a few frames for variety
        img_bgr = demo_frame
        is_demo = True
    else:
        file_bytes = np.frombuffer(uploaded_img.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error("Could not decode the uploaded image. Please try a different file.")
            st.stop()
        is_demo = False

    # Run pipeline
    with st.spinner("Running ADB pipeline…"):
        result: FrameResult = runner.run_frame(img_bgr)
        annotated = compose_annotated_frame(img_bgr, result)

    # ── Layout: image left, metrics right ─────────────────────────────────
    col_img, col_metrics = st.columns([3, 2], gap="medium")

    with col_img:
        st.markdown("<p class='section-header'>Annotated Frame</p>", unsafe_allow_html=True)
        # Convert BGR → RGB for Streamlit
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            use_container_width=True,
            caption=("⚠️ SYNTHETIC DEMO FRAME" if is_demo else uploaded_img.name)
            + (f"  |  Processed in {result.processing_time_ms:.1f} ms"),
        )

    with col_metrics:
        # ── Beam mode badge ────────────────────────────────────────────────
        st.markdown("<p class='section-header'>Current Beam Mode</p>", unsafe_allow_html=True)
        icon = _BEAM_ICONS.get(result.beam_mode, "💡")
        mode_clean = result.beam_mode.replace("_", " ")
        st.markdown(
            f"<div class='beam-badge {result.beam_mode}'>{icon} {mode_clean}</div>",
            unsafe_allow_html=True,
        )
        st.caption(_BEAM_DESC.get(result.beam_mode, ""))

        # ── Quick metrics ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Objects", len(result.detections))
        m2.metric("Weather", result.weather.upper())
        m3.metric("Latency", f"{result.processing_time_ms:.0f} ms")

        # ── Per-zone risk chart ────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>Per-Zone Glare Risk & LED Brightness</p>", unsafe_allow_html=True)
        fig_bar = risk_bar_figure(result.zone_risks, result.zone_brightness)
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        else:
            # Fallback to native bar chart
            import pandas as pd
            df_risk = pd.DataFrame({
                "Risk (%)": result.zone_risks,
                "Brightness (%)": [b / 255.0 * 100 for b in result.zone_brightness],
            }, index=[f"Z{i}" for i in range(result.n_zones)])
            st.bar_chart(df_risk, height=200)

    # ── Explainability panel ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Explainability — Highest Risk Zone")

    if result.detections:
        highest = max(result.detections, key=lambda d: d.risk_score)
        zone_max_risk = max(result.zone_risks)
        zone_max_idx = result.zone_risks.index(zone_max_risk)

        xcol1, xcol2 = st.columns(2, gap="medium")

        with xcol1:
            st.markdown(f"<div class='adb-card'>", unsafe_allow_html=True)
            st.markdown(f"**Highest-risk object:** `ID{highest.track_id}` — **{highest.cls_name}**")
            details = {
                "Estimated Distance": f"{highest.distance_m:.1f} m",
                "Confidence": f"{highest.conf:.0%}",
                "Lane Position": highest.lane_position,
                "Assigned Zone": f"Z{highest.zone_idx}",
                "Glare Risk Score": f"{highest.risk_score:.1f} / 100",
            }
            for k, v in details.items():
                st.markdown(
                    f"<div class='explain-row'>"
                    f"<span class='explain-key'>{k}</span>"
                    f"<span class='explain-val'>{v}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with xcol2:
            if highest.feature_contributions:
                st.markdown(f"<div class='adb-card'>", unsafe_allow_html=True)
                st.markdown("**Top contributing factors:**")
                sorted_contribs = sorted(
                    highest.feature_contributions.items(),
                    key=lambda x: abs(float(x[1])) if isinstance(x[1], (int, float)) else 0,
                    reverse=True,
                )
                for k, v in sorted_contribs[:6]:
                    display_v = f"{float(v):.3f}" if isinstance(v, (int, float)) else str(v)
                    st.markdown(
                        f"<div class='explain-row'>"
                        f"<span class='explain-key'>{k.replace('_', ' ').title()}</span>"
                        f"<span class='explain-val'>{display_v}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Feature contributions unavailable for this backend.")

        # Zone summary
        st.markdown(
            f"<div class='adb-card' style='margin-top:8px;'>"
            f"<b>Highest-risk zone:</b> Z{zone_max_idx} — "
            f"<span style='color:#f85149;'>{zone_max_risk:.1f}% risk</span>, "
            f"LED brightness target: <span style='color:#58a6ff;'>{result.zone_brightness[zone_max_idx]} / 255 PWM</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No objects detected in this frame — no explainability data available.")

    # ── All detections table ───────────────────────────────────────────────
    if result.detections:
        with st.expander("📋 All Detections", expanded=False):
            import pandas as pd
            rows = [
                {
                    "Track ID": d.track_id,
                    "Class": d.cls_name,
                    "Conf": f"{d.conf:.0%}",
                    "Distance (m)": f"{d.distance_m:.1f}",
                    "Lane": d.lane_position,
                    "Zone": f"Z{d.zone_idx}",
                    "Risk (%)": f"{d.risk_score:.1f}",
                }
                for d in result.detections
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    # ── System Health Panel — separate from glare-risk explainability ─────────
    st.markdown("---")
    if _HAS_HEALTH_PANEL:
        render_health_panel(
            findings=getattr(result, 'health_findings', []),
            histories=getattr(result, 'health_histories', {}),
            frame_idx=getattr(runner._health_monitor, 'frame', 0) if hasattr(runner, '_health_monitor') and runner._health_monitor else 0,
        )
    else:
        st.info("🔧 System Health panel not available (diagnostics module not installed).")


# ══════════════════════════════════════════════════════════════════════════════
with tab_video:
    uploaded_vid = st.file_uploader(
        "Upload a dashcam video clip (MP4 / MOV / AVI)",
        type=["mp4", "mov", "avi", "mkv"],
        key="vid_uploader",
    )

    if uploaded_vid is None:
        st.markdown(
            "<div class='mock-banner'>ℹ️ Upload a video clip to see risk-over-time analysis.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("""
        **What you'll see after uploading:**
        - 🎞️ Annotated frame carousel (bounding boxes, zone bars, beam badge)
        - 📈 Per-zone glare risk over time (interactive Plotly)
        - 🔀 Beam mode transitions over time
        - ⚡ Per-frame processing latency
        """)
    else:
        # Write to temp file so OpenCV can open it by path
        suffix = Path(uploaded_vid.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_vid.read())
            tmp_path = tmp.name

        st.info(
            f"📂 `{uploaded_vid.name}` uploaded. "
            f"Sampling every **{sample_n}** frame(s). Processing…"
        )

        process_and_render_video(
            video_path=tmp_path,
            runner=runner,
            sample_n=sample_n,
            key_prefix="vid_upload",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Webcam Video Capture
# ══════════════════════════════════════════════════════════════════════════════
with tab_webcam:
    st.markdown("### 📹 Webcam Video Capture & Live Feed")
    st.caption(
        "Capture a short live webcam video clip or stream live video to run through the frame-by-frame ADB pipeline."
    )

    mode_options = [
        "Option B: Record then Analyze (Primary & Reliable)",
        "Option A: Live WebRTC Stream (Experimental)",
    ]
    selected_webcam_mode = st.radio(
        "Webcam Mode:",
        mode_options,
        index=0,
        horizontal=True,
        help="Option B records a clip in-browser and processes it through the pipeline. Option A streams live via WebRTC.",
    )

    st.markdown(
        "<div class='mock-banner'>ℹ️ <b>Camera Access:</b> Your browser will prompt for camera permission. "
        "Recorded video clips are processed frame-by-frame using the exact same ADB pipeline as uploaded MP4 files.</div>",
        unsafe_allow_html=True,
    )

    if "Option B" in selected_webcam_mode:
        webcam_clip_path = render_webcam_recorder(key="webcam_recorder_input")

        if webcam_clip_path:
            st.markdown("---")
            st.markdown("### 📊 Recorded Clip Analysis")
            process_and_render_video(
                video_path=webcam_clip_path,
                runner=runner,
                sample_n=sample_n,
                key_prefix="webcam_clip",
            )
        else:
            st.info("💡 Select duration and click **Start Recording** above to capture a clip and analyze per-zone risk over time.")

    else:
        if is_webrtc_available():
            st.markdown("#### ⚡ Live WebRTC Streamer")
            st.caption("Frame-skipping is applied to maintain real-time performance.")
            render_webrtc_live_streamer(runner, frame_skip=max(1, sample_n))
        else:
            st.error(
                "⚠️ Live webcam capture unavailable in this environment — please use the video upload option instead. "
                "(Module `streamlit-webrtc` is not installed or WebRTC connection failed)."
            )



# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#8b949e; font-size:0.75rem;'>"
    "Smart Adaptive Headlight (ADB) Demo Dashboard · "
    "No hardware required · SerialBridge always mocked · "
    "All perception mocked when 'Force Mock Mode' is enabled"
    "</p>",
    unsafe_allow_html=True,
)
