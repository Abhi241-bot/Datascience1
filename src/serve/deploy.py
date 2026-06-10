"""Standalone single-container app for the public demo (Hugging Face Space).

Serves the inference API AND the drift dashboard on one port, loading the
model baked in at build time (no MLflow server required). Adds a one-click
``/simulate`` so a visitor can trigger the drift event and watch the monitor
flip — the whole recruiter story in one URL.

Run with:  uvicorn src.serve.deploy:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import json

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from src import config
from src.data.drifted import apply_drift
from src.data.preprocess import engineer_features
from src.monitor import drift, live_log
from src.serve.schemas import PredictionResponse, TransactionRequest, request_to_frame

_MODEL = mlflow.sklearn.load_model(str(config.DEPLOY_MODEL_DIR))
_METRICS_PATH = config.DEPLOY_MODEL_DIR.parent / "metrics.json"
_METRICS = json.loads(_METRICS_PATH.read_text()) if _METRICS_PATH.exists() else {}
_VERSION = "demo"

app = FastAPI(
    title="Real-Time Fraud Detection — Live Demo",
    description="Inference API + live drift monitoring in one container.",
    version="1.0.0",
)


def _predict_and_log(frame: pd.DataFrame) -> float:
    proba = float(_MODEL.predict_proba(frame)[0, 1])
    live_log.append_prediction(engineer_features(frame).iloc[0].to_dict(), proba)
    return proba


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": True, "live_rows": live_log.count()}


@app.get("/model-info")
def model_info() -> dict:
    return {"model_name": config.REGISTERED_MODEL_NAME, "model_version": _VERSION,
            "metrics": _METRICS}


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: TransactionRequest) -> PredictionResponse:
    proba = _predict_and_log(request_to_frame(txn))
    return PredictionResponse(
        fraud_probability=proba,
        label=int(proba >= config.DECISION_THRESHOLD),
        threshold=config.DECISION_THRESHOLD,
        model_version=_VERSION,
    )


@app.post("/simulate")
def simulate(n: int = 400) -> JSONResponse:
    """Replay in-distribution traffic, then a shifted batch, to populate the
    live window so the monitor flips to 'drift detected'."""
    live_log.reset_live_log()
    pool = (pd.read_csv(config.TRAIN_CSV)[config.FEATURE_COLS]
            .sample(n=2 * n, random_state=config.RANDOM_SEED).reset_index(drop=True))
    genuine, drifted = pool.iloc[:n], apply_drift(pool.iloc[n:])
    for _, row in pd.concat([genuine, drifted]).iterrows():
        _predict_and_log(pd.DataFrame([row[config.FEATURE_COLS].to_dict()]))
    return JSONResponse(drift.compute_summary())


@app.get("/drift-summary")
def drift_summary() -> JSONResponse:
    return JSONResponse(drift.compute_summary())


@app.get("/monitoring", response_class=HTMLResponse)
def monitoring() -> str:
    report, _ = drift.build_report()
    path = drift.save_report(report)
    from pathlib import Path
    return Path(path).read_text(encoding="utf-8")


@app.post("/reset")
def reset() -> dict:
    live_log.reset_live_log()
    return {"status": "reset"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    s = drift.compute_summary()
    flag = "🚨 DRIFT DETECTED" if s["drift_detected"] else "✅ No drift"
    color = "#c0392b" if s["drift_detected"] else "#27ae60"
    pr_auc = _METRICS.get("pr_auc")
    pr_auc_str = f"{pr_auc:.3f}" if isinstance(pr_auc, (int, float)) else "n/a"
    return f"""
    <html><head><title>Real-Time Fraud Detection</title>
    <style>
      body{{font-family:system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 16px}}
      .badge{{color:#fff;background:{color};padding:6px 14px;border-radius:6px;font-weight:600}}
      button{{font-size:16px;padding:10px 18px;border:0;border-radius:6px;background:#2c3e50;
        color:#fff;cursor:pointer}}
      a{{color:#2980b9}} code{{background:#f4f4f4;padding:2px 5px;border-radius:4px}}
      .row{{margin:18px 0}}
    </style></head>
    <body>
      <h1>Real-Time Fraud Detection & Drift Monitoring</h1>
      <p>XGBoost fraud scorer served behind FastAPI, with live Evidently drift
         monitoring and an automated retraining loop. Test PR-AUC <b>{pr_auc_str}</b>
         on a ~0.17%-positive problem.</p>
      <div class="row"><span class="badge">{flag}</span>
         &nbsp; live predictions observed: <b>{s['n_live_rows']}</b>,
         drifted features {s['n_drifted_columns']}/{s['n_columns']}.</div>
      <div class="row">
        <button onclick="sim()">▶ Simulate a drift event</button>
        &nbsp; then open the <a href="/monitoring">live drift report</a>.
      </div>
      <p class="row">Try the API: <a href="/docs">interactive Swagger docs</a>
         (<code>POST /predict</code>).</p>
      <p id="status" style="color:#888"></p>
      <script>
      async function sim(){{
        const el=document.getElementById('status');
        el.textContent='Simulating genuine traffic, then injecting drift…';
        const r=await fetch('/simulate',{{method:'POST'}}); const j=await r.json();
        el.innerHTML='Done — drift_detected='+j.drift_detected+
          ' (share '+j.share_drifted_columns.toFixed(2)+'). '+
          '<a href="/monitoring">Open the drift report »</a>';
      }}
      </script>
    </body></html>
    """
