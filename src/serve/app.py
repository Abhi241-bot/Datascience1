"""Real-time fraud inference API.

Loads the ``Production`` model from the MLflow registry at startup and serves
predictions. A ``/reload`` endpoint lets the closed loop (Phase 5) pick up a
newly promoted model without a restart.

Run with:  uvicorn src.serve.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

from src import config
from src.data.preprocess import engineer_features
from src.monitor import live_log
from src.serve.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionRequest,
    request_to_frame,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("serve")


@dataclass
class ModelHolder:
    pipeline: object | None = None
    version: str = "none"
    stage: str = config.PRODUCTION_STAGE
    pr_auc: str | None = None

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None


MODEL = ModelHolder()


def load_production_model() -> bool:
    """(Re)load the current Production model. Returns True on success."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    versions = client.get_latest_versions(
        config.REGISTERED_MODEL_NAME, stages=[config.PRODUCTION_STAGE]
    )
    if not versions:
        log.warning("no model in '%s' stage yet", config.PRODUCTION_STAGE)
        return False

    mv = versions[0]
    uri = f"models:/{config.REGISTERED_MODEL_NAME}/{mv.version}"
    log.info("loading model %s", uri)
    MODEL.pipeline = mlflow.sklearn.load_model(uri)
    MODEL.version = mv.version
    MODEL.pr_auc = mv.tags.get("pr_auc")
    log.info("loaded fraud-xgb v%s (PR-AUC=%s)", mv.version, MODEL.pr_auc)
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_production_model()
    except Exception as exc:  # don't crash the app; /health will report it
        log.error("startup model load failed: %s", exc)
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud risk scoring served from the MLflow registry.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=MODEL.loaded)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    if not MODEL.loaded:
        raise HTTPException(status_code=503, detail="no model loaded")
    return ModelInfoResponse(
        model_name=config.REGISTERED_MODEL_NAME,
        model_version=MODEL.version,
        stage=MODEL.stage,
        pr_auc=MODEL.pr_auc,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: TransactionRequest) -> PredictionResponse:
    if not MODEL.loaded:
        raise HTTPException(status_code=503, detail="model not loaded")
    frame = request_to_frame(txn)
    proba = float(MODEL.pipeline.predict_proba(frame)[0, 1])
    label = int(proba >= config.DECISION_THRESHOLD)
    # Log the engineered features + prediction so the monitor can detect drift.
    live_log.append_prediction(engineer_features(frame).iloc[0].to_dict(), proba)
    return PredictionResponse(
        fraud_probability=proba,
        label=label,
        threshold=config.DECISION_THRESHOLD,
        model_version=MODEL.version,
    )


@app.post("/reload", response_model=ModelInfoResponse)
def reload() -> ModelInfoResponse:
    """Reload the current Production model (used after a retrain/promote)."""
    if not load_production_model():
        raise HTTPException(status_code=503, detail="no Production model to load")
    return model_info()
