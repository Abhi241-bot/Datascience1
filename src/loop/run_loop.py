"""The closed loop: drift -> retrain -> promote-if-better -> serve.

Run with:  python -m src.loop.run_loop [--force]

Steps (each prints a visible event so the demo is auditable):
  1. Read the live drift summary from the monitor.
  2. If drift is detected, retrain a challenger that has seen the new regime.
  3. FAIR GATE: score the current Production model AND the challenger on the
     *same* current (drift-inclusive) evaluation set, and promote the challenger
     only if its PR-AUC is genuinely higher.
  4. Tell the API to reload, and confirm it now serves the new version.
"""
from __future__ import annotations

import sys

import mlflow
import mlflow.sklearn
import pandas as pd
import requests
from mlflow.tracking import MlflowClient

from src import config
from src.data.drifted import augment_with_drift
from src.registry.promote import register_and_promote
from src.train.evaluate import compute_metrics
from src.train.train import train

_LOG: list[str] = []


def _event(msg: str) -> None:
    print(f"[loop] {msg}")
    _LOG.append(msg)


def _current_eval_set() -> tuple[pd.DataFrame, pd.Series]:
    """Held-out test data + shifted copies = the current production reality."""
    aug = augment_with_drift(pd.read_csv(config.TEST_CSV))
    return aug[config.FEATURE_COLS], aug[config.TARGET_COL].astype(int)


def _pr_auc(model, x, y) -> float:
    return compute_metrics(y, model.predict_proba(x)[:, 1])["pr_auc"]


def _production_version(client: MlflowClient):
    versions = client.get_latest_versions(
        config.REGISTERED_MODEL_NAME, stages=[config.PRODUCTION_STAGE]
    )
    return versions[0] if versions else None


def run(force: bool = False) -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print("=" * 64)
    print("CLOSED LOOP: drift -> retrain -> promote-if-better -> serve")
    print("=" * 64)

    # --- Step 1: check drift --------------------------------------------- #
    summary = requests.get(f"{config.MONITOR_URL}/drift-summary", timeout=60).json()
    drifted = bool(summary.get("drift_detected"))
    _event(f"STEP 1  drift check: drift_detected={drifted} "
           f"(share={summary.get('share_drifted_columns'):.2f}, "
           f"{summary.get('n_drifted_columns')}/{summary.get('n_columns')} features)")
    if not drifted and not force:
        _event("no drift above threshold — loop ends (nothing to retrain)")
        return

    # --- Step 2: retrain a challenger on the new regime ------------------ #
    _event("STEP 2  drift confirmed -> retraining challenger on current data")
    challenger_run = train(include_drift=True)
    _event(f"STEP 2  challenger trained: run_id={challenger_run}")

    # --- Step 3: FAIR GATE — score both on the same current eval set ----- #
    x_eval, y_eval = _current_eval_set()
    prod_mv = _production_version(client)

    challenger = mlflow.sklearn.load_model(f"runs:/{challenger_run}/model")
    chal_prauc = _pr_auc(challenger, x_eval, y_eval)

    if prod_mv is None:
        _event("STEP 3  no current Production model — challenger wins by default")
        prod_prauc, promote_it = -1.0, True
    else:
        prod_model = mlflow.sklearn.load_model(
            f"models:/{config.REGISTERED_MODEL_NAME}/{config.PRODUCTION_STAGE}"
        )
        prod_prauc = _pr_auc(prod_model, x_eval, y_eval)
        promote_it = chal_prauc > prod_prauc + config.PROMOTION_MIN_IMPROVEMENT
        _event(f"STEP 3  fair eval on current data ({len(y_eval)} rows): "
               f"Production v{prod_mv.version} PR-AUC={prod_prauc:.4f}  vs  "
               f"challenger PR-AUC={chal_prauc:.4f}")

    if not promote_it:
        _event("STEP 3  challenger NOT better -> keeping current Production (gate held)")
        _print_log()
        return

    # --- Step 4: promote + serve ---------------------------------------- #
    new_version = register_and_promote(client, challenger_run, chal_prauc)
    _event(f"STEP 3  challenger better -> PROMOTED {config.REGISTERED_MODEL_NAME} "
           f"v{new_version} to Production")

    resp = requests.post(f"{config.API_URL}/reload", timeout=120)
    resp.raise_for_status()
    served = resp.json().get("model_version")
    _event(f"STEP 4  API reloaded -> now serving v{served}")
    if served == new_version:
        _event("STEP 4  VERIFIED: API is serving the freshly promoted model ✔")
    else:
        _event(f"STEP 4  WARNING: API serves v{served} but promoted v{new_version}")

    _print_log()


def _print_log() -> None:
    print("\n" + "-" * 64)
    print("CLOSED-LOOP EVENT LOG")
    print("-" * 64)
    for i, msg in enumerate(_LOG, 1):
        print(f"  {i:>2}. {msg}")


if __name__ == "__main__":
    run(force="--force" in sys.argv)
