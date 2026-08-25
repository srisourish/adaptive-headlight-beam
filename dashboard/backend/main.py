"""
Dashboard backend for smart-adaptive-headlight.

FastAPI app exposing WebSocket streaming:
  - Live camera frame (JPEG)
  - Per-zone risk array
  - Current beam mode
  - Explainability log entries
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# FastAPI imports
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    print("[Dashboard] FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

app = FastAPI(title="Smart Adaptive Headlight Dashboard")

# Serve frontend static files
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Shared State (populated by pipeline or mock) ───────────────
class DashboardState:
    """Thread-safe shared state for the dashboard."""

    def __init__(self) -> None:
        self.frame: np.ndarray | None = None
        self.zone_risks: list[float] = [0.0] * 8
        self.zone_brightnesses: list[int] = [255] * 8
        self.beam_mode: str = "LOW_BEAM"
        self.explainability: list[dict] = []
        self.fps: float = 0.0
        self.timestamp: float = time.time()

    def to_json(self) -> dict:
        frame_b64 = ""
        if self.frame is not None:
            _, buf = cv2.imencode(".jpg", self.frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        return {
            "frame": frame_b64,
            "zone_risks": self.zone_risks,
            "zone_brightnesses": self.zone_brightnesses,
            "beam_mode": self.beam_mode,
            "explainability": self.explainability[-5:],
            "fps": round(self.fps, 1),
            "timestamp": self.timestamp,
        }


state = DashboardState()


def update_state(
    frame: np.ndarray | None = None,
    zone_risks: list[float] | None = None,
    zone_brightnesses: list[int] | None = None,
    beam_mode: str | None = None,
    explainability: list[dict] | None = None,
    fps: float | None = None,
) -> None:
    """Update dashboard state from the pipeline."""
    if frame is not None:
        state.frame = frame
    if zone_risks is not None:
        state.zone_risks = zone_risks
    if zone_brightnesses is not None:
        state.zone_brightnesses = zone_brightnesses
    if beam_mode is not None:
        state.beam_mode = beam_mode
    if explainability is not None:
        state.explainability = explainability
    if fps is not None:
        state.fps = fps
    state.timestamp = time.time()


# ── Mock data generator for standalone testing ──────────────────
async def mock_data_loop() -> None:
    """Generate synthetic dashboard data when running standalone."""
    t = 0
    while True:
        # Synthetic frame
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[180:, :] = (40, 40, 40)
        cx = int(320 + 200 * np.sin(t * 0.05))
        cv2.circle(frame, (cx, 200), 15, (200, 220, 255), -1)
        cv2.putText(frame, f"t={t}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Synthetic risks
        risks = [float(50 + 40 * np.sin(t * 0.03 + i * 0.5)) for i in range(8)]
        risks = [max(0, min(100, r)) for r in risks]

        from decision.zone_mapper import ZoneMapper
        zm = ZoneMapper()
        brightnesses = zm.zone_brightnesses(risks)

        mode = "MATRIX_PARTIAL" if any(r > 60 for r in risks) else "HIGH_BEAM"

        update_state(
            frame=frame, zone_risks=risks,
            zone_brightnesses=brightnesses,
            beam_mode=mode, fps=30.0,
            explainability=[{
                "zone_id": 3, "risk_score": risks[3],
                "beam_mode": mode,
                "top_features": [
                    {"name": "proximity", "value": 0.22},
                    {"name": "lane_relevance", "value": 0.18},
                ],
                "object_type": "car", "distance_m": 25.0,
                "lane_position": "oncoming",
            }],
        )
        t += 1
        await asyncio.sleep(1 / 30)


# ── Routes ──────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard frontend not found</h1>")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Stream dashboard data to the frontend via WebSocket."""
    await ws.accept()
    try:
        while True:
            data = state.to_json()
            await ws.send_text(json.dumps(data))
            await asyncio.sleep(1 / 15)  # 15 FPS to client
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.get("/api/state")
async def get_state():
    """REST endpoint for current state (without frame)."""
    data = state.to_json()
    data.pop("frame", None)
    return data


# ── Startup ─────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Start mock data loop if running standalone."""
    asyncio.create_task(mock_data_loop())


if __name__ == "__main__":
    print("[Dashboard] Starting at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
