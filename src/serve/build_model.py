"""Build a standalone model artifact for deployment (no MLflow server).

Runs at Docker build time for the Hugging Face Space: downloads the data,
preprocesses it (which also writes the reference window), fits the pipeline,
and saves it to ``config.DEPLOY_MODEL_DIR`` so the deploy app can load it
offline.

Run with:  python -m src.serve.build_model
"""
from __future__ import annotations

import json

import pickle
import pandas as pd

from src import config
from src.data import download, preprocess
from src.train.evaluate import compute_metrics, format_metrics
from src.train.train import build_pipeline


def main() -> None:
    download.download()
    preprocess.preprocess()

    train = pd.read_csv(config.TRAIN_CSV)
    test = pd.read_csv(config.TEST_CSV)
    x_train, y_train = train[config.FEATURE_COLS], train[config.TARGET_COL].astype(int)
    x_test, y_test = test[config.FEATURE_COLS], test[config.TARGET_COL].astype(int)

    n_pos = int(y_train.sum())
    scale_pos_weight = (len(y_train) - n_pos) / max(n_pos, 1)
    pipeline = build_pipeline(scale_pos_weight)
    pipeline.fit(x_train, y_train)

    metrics = compute_metrics(y_test, pipeline.predict_proba(x_test)[:, 1])
    print("[build_model] test metrics:\n" + format_metrics(metrics))

    config.DEPLOY_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.DEPLOY_MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    (config.DEPLOY_MODEL_DIR.parent / "metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    print(f"[build_model] saved model to {config.DEPLOY_MODEL_DIR / 'model.pkl'}")


if __name__ == "__main__":
    main()
