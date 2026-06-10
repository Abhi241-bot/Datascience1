"""Single source of truth for all paths, thresholds, and hyperparameters.

No magic numbers anywhere else in the codebase — import from here.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# When running in Docker the working dir is /app; locally it is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_REFERENCE = DATA_DIR / "reference"
DATA_LIVE = DATA_DIR / "live"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _p in (DATA_RAW, DATA_PROCESSED, DATA_REFERENCE, DATA_LIVE, REPORTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# Live prediction log (written by the API, read by the monitor) and the
# generated Evidently report.
LIVE_LOG_CSV = DATA_LIVE / "live_log.csv"
DRIFT_REPORT_HTML = REPORTS_DIR / "drift_report.html"

# Standalone deployment (Hugging Face Space): model saved here at build time so
# the container serves without an MLflow server.
DEPLOY_MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"

RAW_CSV = DATA_RAW / "creditcard.csv"
REFERENCE_CSV = DATA_REFERENCE / "reference.csv"

TRAIN_CSV = DATA_PROCESSED / "train.csv"
VAL_CSV = DATA_PROCESSED / "val.csv"
TEST_CSV = DATA_PROCESSED / "test.csv"

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
# Public mirror of the ULB "Credit Card Fraud Detection" dataset, hosted by
# TensorFlow. No credentials required — reproducible from a clean clone.
DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
EXPECTED_ROWS = 284_807  # sanity check after download

TARGET_COL = "Class"
TIME_COL = "Time"
AMOUNT_COL = "Amount"
# V1..V28 are anonymised PCA components in the ULB dataset.
PCA_COLS = [f"V{i}" for i in range(1, 29)]

# Final feature columns fed to the model (Amount is log-transformed in the
# pipeline; Time is used only to order the split, not as a model feature).
FEATURE_COLS = PCA_COLS + [AMOUNT_COL]

# --------------------------------------------------------------------------- #
# Split (time-ordered, not random — mirrors how the model meets new data)
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test = remainder (0.15)

# The reference window for drift monitoring = the training distribution.
REFERENCE_SAMPLE_SIZE = 10_000

# --------------------------------------------------------------------------- #
# Model hyperparameters (XGBoost)
# --------------------------------------------------------------------------- #
# scale_pos_weight is computed at train time from the actual class balance.
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

# Decision threshold for converting fraud probability -> label.
DECISION_THRESHOLD = 0.5

# --------------------------------------------------------------------------- #
# MLflow
# --------------------------------------------------------------------------- #
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = "fraud-detection"
REGISTERED_MODEL_NAME = "fraud-xgb"
PRODUCTION_STAGE = "Production"

# --------------------------------------------------------------------------- #
# Drift / retraining thresholds (used in Phases 4 & 5)
# --------------------------------------------------------------------------- #
# Share of features that must drift before we flag dataset drift.
DRIFT_SHARE_THRESHOLD = 0.5
# Number of most-recent live predictions compared against the reference window.
LIVE_WINDOW_SIZE = 2000

# Magnitude of the deliberate drift event. Used in two places that MUST agree:
# the live simulator (shifts inference traffic) and the retrain/eval data
# builder (shifts labeled data so the new regime can be learned & scored).
DRIFT_FEATURE_SHIFT = 5.0
DRIFT_AMOUNT_MULTIPLIER = 12.0
# A new model is promoted only if its PR-AUC beats Production by this margin.
PROMOTION_MIN_IMPROVEMENT = 0.0

# --------------------------------------------------------------------------- #
# Service URLs (used by the simulator and the closed-loop runner)
# --------------------------------------------------------------------------- #
API_URL = os.getenv("API_URL", "http://localhost:8000")
MONITOR_URL = os.getenv("MONITOR_URL", "http://localhost:8050")
