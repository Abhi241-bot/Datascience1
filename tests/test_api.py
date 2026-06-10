"""API tests. A dummy model is injected so the schema/contract can be tested
without a running MLflow registry (CI has no model store)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import config
from src.serve import app as app_module

# TestClient without a context manager => the lifespan model-load (which would
# hit MLflow) is not triggered; we inject a deterministic dummy instead.
client = TestClient(app_module.app)


class _DummyModel:
    def predict_proba(self, frame):
        # Always returns P(fraud)=0.8 for a single-row request.
        return np.array([[0.2, 0.8]])


@pytest.fixture(autouse=True)
def _inject_model():
    app_module.MODEL.pipeline = _DummyModel()
    app_module.MODEL.version = "test"
    app_module.MODEL.pr_auc = "0.9"
    yield
    app_module.MODEL.pipeline = None


def _valid_payload() -> dict:
    payload = {col: 0.0 for col in config.PCA_COLS}
    payload[config.AMOUNT_COL] = 100.0
    return payload


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    assert r.json()["model_version"] == "test"


def test_predict_returns_valid_schema():
    r = client.post("/predict", json=_valid_payload())
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"fraud_probability", "label", "threshold", "model_version"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["label"] in (0, 1)
    assert body["label"] == 1  # 0.8 >= 0.5 threshold
    assert body["model_version"] == "test"


def test_predict_rejects_missing_field():
    bad = _valid_payload()
    del bad["V1"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422  # pydantic validation error
