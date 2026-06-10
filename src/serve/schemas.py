"""Pydantic request/response models for the inference API.

The transaction model is built dynamically from ``config.PCA_COLS`` so the 28
anonymised features stay in sync with the rest of the pipeline (DRY). A real
fraud row from the held-out test set is embedded as the Swagger example so
``/docs`` works out of the box.
"""
from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, create_model

from src import config

# A genuine fraud transaction from the test split — makes /docs immediately
# runnable and demonstrates a positive prediction.
_EXAMPLE = {
    "V1": -5.488, "V2": 3.330, "V3": -5.996, "V4": 3.602, "V5": -2.024,
    "V6": -1.737, "V7": -4.397, "V8": 0.228, "V9": -1.676, "V10": -3.992,
    "V11": 3.737, "V12": -6.150, "V13": 0.289, "V14": -8.761, "V15": 2.345,
    "V16": -5.146, "V17": -6.293, "V18": -2.137, "V19": 2.995, "V20": -0.551,
    "V21": 1.720, "V22": 0.343, "V23": 0.134, "V24": 0.833, "V25": -0.840,
    "V26": 0.502, "V27": -1.937, "V28": 1.521, "Amount": 0.01,
}

# Field definitions: V1..V28 (anonymised PCA components) + Amount.
_fields: dict = {
    col: (float, Field(..., description=f"Anonymised PCA component {col}"))
    for col in config.PCA_COLS
}
_fields[config.AMOUNT_COL] = (
    float,
    Field(..., ge=0, description="Transaction amount (>= 0)"),
)

TransactionRequest = create_model(  # type: ignore[call-overload]
    "TransactionRequest",
    **_fields,
    __config__=type("Cfg", (), {"json_schema_extra": {"example": _EXAMPLE}}),
)


def request_to_frame(req: "TransactionRequest") -> pd.DataFrame:
    """Convert a validated request into the single-row frame the pipeline expects."""
    row = {col: getattr(req, col) for col in config.FEATURE_COLS}
    return pd.DataFrame([row], columns=config.FEATURE_COLS)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    fraud_probability: float = Field(..., description="P(fraud) in [0, 1]")
    label: int = Field(..., description="1 = fraud, 0 = legitimate")
    threshold: float = Field(..., description="Decision threshold applied")
    model_version: str = Field(..., description="MLflow registry version served")


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    stage: str
    pr_auc: str | None = None
