/**
 * Smart Adaptive Headlight — Dashboard Frontend
 *
 * Connects to the FastAPI WebSocket and renders:
 *   - Live video feed
 *   - Per-zone brightness strip
 *   - Risk score bars
 *   - Decision explainability panel
 */

(function () {
    "use strict";

    const WS_URL = `ws://${window.location.host}/ws`;
    const ZONE_COUNT = 8;
    const RECONNECT_DELAY_MS = 2000;

    // DOM refs
    const videoFeed = document.getElementById("video-feed");
    const beamMode = document.getElementById("beam-mode");
    const connStatus = document.getElementById("connection-status");
    const fpsCounter = document.getElementById("fps-counter");
    const zoneStrip = document.getElementById("zone-strip");
    const riskBars = document.getElementById("risk-bars");
    const explainContent = document.getElementById("explain-content");

    let ws = null;
    let frameCount = 0;
    let lastFpsTime = performance.now();

    // ── Initialize UI ──────────────────────────────────────────

    function initZoneStrip() {
        zoneStrip.innerHTML = "";
        for (let i = 0; i < ZONE_COUNT; i++) {
            const bar = document.createElement("div");
            bar.className = "zone-bar";
            bar.id = `zone-${i}`;
            bar.innerHTML = `<span>${i}</span>`;
            zoneStrip.appendChild(bar);
        }
    }

    function initRiskBars() {
        riskBars.innerHTML = "";
        for (let i = 0; i < ZONE_COUNT; i++) {
            const row = document.createElement("div");
            row.className = "risk-row";
            row.innerHTML = `
                <span class="risk-label">Zone ${i}</span>
                <div class="risk-track">
                    <div class="risk-fill" id="risk-fill-${i}" style="width: 0%; background: #34d399;"></div>
                </div>
                <span class="risk-value" id="risk-val-${i}">0</span>
            `;
            riskBars.appendChild(row);
        }
    }

    // ── Update UI ──────────────────────────────────────────────

    function riskToColor(risk) {
        if (risk < 30) return "#34d399";      // green
        if (risk < 60) return "#fbbf24";      // amber
        return "#ef4444";                     // red
    }

    function brightnessToColor(b) {
        const ratio = b / 255;
        const r = Math.round(200 + 55 * ratio);
        const g = Math.round(180 + 40 * ratio);
        const bl = Math.round(80 + 60 * ratio);
        return `rgba(${r}, ${g}, ${bl}, ${0.3 + 0.7 * ratio})`;
    }

    function updateFrame(frameB64) {
        if (frameB64) {
            videoFeed.src = "data:image/jpeg;base64," + frameB64;
        }
        frameCount++;
        const now = performance.now();
        if (now - lastFpsTime > 1000) {
            const fps = (frameCount / ((now - lastFpsTime) / 1000)).toFixed(1);
            fpsCounter.textContent = fps + " FPS";
            frameCount = 0;
            lastFpsTime = now;
        }
    }

    function updateBeamMode(mode) {
        beamMode.textContent = mode.replace(/_/g, " ");
        beamMode.className = "mode-badge";
        if (mode.includes("HIGH")) beamMode.classList.add("high");
        else if (mode.includes("LOW")) beamMode.classList.add("low");
        else if (mode.includes("MEDIUM")) beamMode.classList.add("medium");
        else beamMode.classList.add("matrix");
    }

    function updateZones(brightnesses) {
        for (let i = 0; i < ZONE_COUNT; i++) {
            const bar = document.getElementById(`zone-${i}`);
            if (bar) {
                const b = brightnesses[i] || 0;
                bar.style.background = brightnessToColor(b);
                if (b > 128) {
                    bar.style.boxShadow = `0 0 12px ${brightnessToColor(b)}`;
                } else {
                    bar.style.boxShadow = "none";
                }
                bar.querySelector("span").textContent = b;
            }
        }
    }

    function updateRisks(risks) {
        for (let i = 0; i < ZONE_COUNT; i++) {
            const fill = document.getElementById(`risk-fill-${i}`);
            const val = document.getElementById(`risk-val-${i}`);
            if (fill && val) {
                const r = Math.round(risks[i] || 0);
                fill.style.width = r + "%";
                fill.style.background = riskToColor(r);
                val.textContent = r;
                val.style.color = riskToColor(r);
            }
        }
    }

    function updateExplainability(entries) {
        if (!entries || entries.length === 0) return;
        let html = "";
        for (const e of entries) {
            const features = (e.top_features || [])
                .map(f => {
                    const pct = Math.round(Math.abs(f.value) * 100);
                    return `
                        <div class="feature-row">
                            <span class="feature-name">${f.name}</span>
                            <div class="feature-bar-track">
                                <div class="feature-bar-fill" style="width: ${pct}%"></div>
                            </div>
                        </div>`;
                })
                .join("");

            html += `
                <div class="explain-entry">
                    <div class="explain-header">
                        <span class="obj-type">${e.object_type || "—"} (zone ${e.zone_id})</span>
                        <span class="distance">${e.distance_m ? e.distance_m.toFixed(0) + "m" : ""} · ${e.lane_position || ""}</span>
                    </div>
                    <div class="explain-features">${features}</div>
                </div>`;
        }
        explainContent.innerHTML = html;
    }

    // ── WebSocket ──────────────────────────────────────────────

    function connect() {
        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            connStatus.textContent = "Connected";
            connStatus.className = "status-badge connected";
        };

        ws.onclose = () => {
            connStatus.textContent = "Disconnected";
            connStatus.className = "status-badge disconnected";
            setTimeout(connect, RECONNECT_DELAY_MS);
        };

        ws.onerror = () => {
            ws.close();
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                updateFrame(data.frame);
                updateBeamMode(data.beam_mode || "LOW_BEAM");
                updateZones(data.zone_brightnesses || []);
                updateRisks(data.zone_risks || []);
                updateExplainability(data.explainability || []);
            } catch (err) {
                console.error("Parse error:", err);
            }
        };
    }

    // ── Boot ───────────────────────────────────────────────────

    initZoneStrip();
    initRiskBars();
    connect();
})();
