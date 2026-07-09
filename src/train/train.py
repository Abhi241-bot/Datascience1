"""Train an XGBoost fraud classifier and log everything to MLflow.

Logs params, imbalance-aware metrics, and the full sklearn pipeline
(feature engineering + scaling + model) as a single artifact so serving is
train/serve consistent.

Run with:  python -m src.train.train
"""
from __future__ import annotations

import sys

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from src import config
from src.data.preprocess import engineer_features
from src.train.evaluate import compute_metrics, format_metrics


def _load_xy(path, include_drift: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if include_drift:
        # Append labeled copies of the shifted regime (the retrain scenario).
        from src.data.drifted import augment_with_drift
        df = augment_with_drift(df)
    x = df[config.FEATURE_COLS]
    y = df[config.TARGET_COL].astype(int)
    return x, y


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("engineer", FunctionTransformer(engineer_features, validate=False)),
            ("scale", StandardScaler()),
            (
                "clf",
                XGBClassifier(scale_pos_weight=scale_pos_weight, **config.XGB_PARAMS),
            ),
        ]
    )


def train(include_drift: bool = False) -> str:
    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    if include_drift:
        print("[train] retrain mode: including labeled drifted-regime data")
    x_train, y_train = _load_xy(config.TRAIN_CSV, include_drift)
    x_val, y_val = _load_xy(config.VAL_CSV, include_drift)
    x_test, y_test = _load_xy(config.TEST_CSV, include_drift)

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"[train] class balance: {n_pos} fraud / {n_neg} legit "
          f"-> scale_pos_weight={scale_pos_weight:.1f}")

    pipeline = build_pipeline(scale_pos_weight)

    with mlflow.start_run() as run:
        print(f"[train] MLflow run_id={run.info.run_id}")
        mlflow.set_tag("trained_on", "original+drift" if include_drift else "original")
        pipeline.fit(x_train, y_train)

        # Evaluate on val and test.
        val_metrics = compute_metrics(y_val, pipeline.predict_proba(x_val)[:, 1])
        test_metrics = compute_metrics(y_test, pipeline.predict_proba(x_test)[:, 1])
        print("[train] validation:\n" + format_metrics(val_metrics))
        print("[train] test:\n" + format_metrics(test_metrics))

        # Log params.
        mlflow.log_params(config.XGB_PARAMS)
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 2))
        mlflow.log_param("decision_threshold", config.DECISION_THRESHOLD)
        mlflow.log_param("n_train", len(y_train))
        mlflow.log_param("train_fraud_rate", round(n_pos / len(y_train), 6))

        # Log metrics (flat keys so they sort nicely in the MLflow UI).
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        # Primary selection metric, unprefixed, for promote.py to query on.
        mlflow.log_metric("pr_auc", val_metrics["pr_auc"])

        # Log the full pipeline with a signature + input example.
        example = x_train.head(2)
        signature = infer_signature(example, pipeline.predict_proba(example)[:, 1])
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=signature,
            input_example=example,
        )
        print(f"[train] logged model artifact (val PR-AUC={val_metrics['pr_auc']:.4f})")
        return run.info.run_id


if __name__ == "__main__":
    run_id = train(include_drift="--include-drift" in sys.argv)
    print(f"[train] done. run_id={run_id}")
    sys.exit(0)
