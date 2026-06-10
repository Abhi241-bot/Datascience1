"""Live monitoring dashboard.

Serves a freshly-generated Evidently drift report at ``/monitoring`` (it
recomputes on every request, so it reflects new live data as the simulator
streams it) and a machine-readable summary at ``/drift-summary``.

Run with:  uvicorn src.monitor.dashboard:app --host 0.0.0.0 --port 8050
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from src.monitor import drift, live_log

app = FastAPI(title="Fraud Drift Monitor", version="1.0.0")

# Auto-refresh so the page visibly flips to "drift detected" during a stream.
_REFRESH_META = '<meta http-equiv="refresh" content="15">'


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "live_rows": live_log.count()}


@app.get("/drift-summary")
def drift_summary() -> JSONResponse:
    return JSONResponse(drift.compute_summary())


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    s = drift.compute_summary()
    flag = "🚨 DRIFT DETECTED" if s["drift_detected"] else "✅ No drift"
    color = "#c0392b" if s["drift_detected"] else "#27ae60"
    return f"""
    <html><head><title>Fraud Drift Monitor</title>{_REFRESH_META}
    <style>body{{font-family:system-ui,sans-serif;margin:40px}}
    .badge{{color:#fff;background:{color};padding:8px 16px;border-radius:6px;
    font-weight:600;font-size:20px}}</style></head>
    <body>
      <h1>Fraud Detection — Live Drift Monitor</h1>
      <p><span class="badge">{flag}</span></p>
      <ul>
        <li>Live predictions observed: <b>{s['n_live_rows']}</b></li>
        <li>Drifted features: <b>{s['n_drifted_columns']} / {s['n_columns']}</b>
            (share {s['share_drifted_columns']:.2f}, threshold {s['threshold']})</li>
      </ul>
      <p><a href="/monitoring">Open full Evidently report &raquo;</a>
         &nbsp;|&nbsp; <a href="/drift-summary">JSON summary</a></p>
      <p style="color:#888">Auto-refreshes every 15s.</p>
    </body></html>
    """


@app.get("/monitoring", response_class=HTMLResponse)
def monitoring() -> str:
    report, _ = drift.build_report()
    path = drift.save_report(report)
    html = Path(path).read_text(encoding="utf-8")
    # Inject auto-refresh into the Evidently report's <head>.
    return html.replace("<head>", "<head>" + _REFRESH_META, 1)


@app.post("/reset")
def reset() -> dict:
    live_log.reset_live_log()
    return {"status": "reset", "live_rows": 0}
