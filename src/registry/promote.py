"""Promote the best training run to the MLflow Model Registry's Production stage.

Phase 2 behaviour (default): find the best run by PR-AUC, register it, and
transition it to ``Production``.

Phase 5 behaviour (``--only-if-better``): promote the challenger ONLY if its
PR-AUC beats the current Production model by ``PROMOTION_MIN_IMPROVEMENT``.
This is the gate that makes the closed loop safe.

Re-running is idempotent: if the best run is already the Production version,
nothing is registered.

Run with:  python -m src.registry.promote [--only-if-better]
"""
from __future__ import annotations

import sys
import time

import mlflow
from mlflow.tracking import MlflowClient

from src import config

PRIMARY_METRIC = "pr_auc"


def _best_run(client: MlflowClient):
    exp = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    if exp is None:
        raise RuntimeError(f"experiment '{config.MLFLOW_EXPERIMENT}' not found — "
                           "run training first")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"attributes.status = 'FINISHED' and metrics.{PRIMARY_METRIC} > 0",
        order_by=[f"metrics.{PRIMARY_METRIC} DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("no finished runs with a pr_auc metric were found")
    return runs[0]


def _production_version(client: MlflowClient):
    try:
        versions = client.get_latest_versions(
            config.REGISTERED_MODEL_NAME, stages=[config.PRODUCTION_STAGE]
        )
    except mlflow.exceptions.MlflowException:
        return None
    return versions[0] if versions else None


def _wait_until_ready(client: MlflowClient, name: str, version: str, timeout: int = 30):
    for _ in range(timeout):
        mv = client.get_model_version(name, version)
        if mv.status == "READY":
            return
        time.sleep(1)
    raise TimeoutError(f"model version {name} v{version} not READY after {timeout}s")


def register_and_promote(client: MlflowClient, run_id: str, pr_auc: float) -> str:
    """Register a specific run's model and transition it to Production.

    Archives whatever was previously in Production. Returns the new version.
    Shared by the best-run promoter below and the closed-loop gate.
    """
    mv = mlflow.register_model(f"runs:/{run_id}/model", config.REGISTERED_MODEL_NAME)
    _wait_until_ready(client, config.REGISTERED_MODEL_NAME, mv.version)
    client.transition_model_version_stage(
        name=config.REGISTERED_MODEL_NAME,
        version=mv.version,
        stage=config.PRODUCTION_STAGE,
        archive_existing_versions=True,
    )
    client.set_model_version_tag(
        config.REGISTERED_MODEL_NAME, mv.version, "pr_auc", f"{pr_auc:.4f}"
    )
    return mv.version


def promote(only_if_better: bool = False) -> bool:
    """Returns True if a new version was promoted, False if skipped."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()

    best = _best_run(client)
    best_prauc = best.data.metrics[PRIMARY_METRIC]
    print(f"[promote] best run {best.info.run_id} PR-AUC={best_prauc:.4f}")

    current = _production_version(client)
    if current is not None:
        cur_prauc = client.get_run(current.run_id).data.metrics.get(PRIMARY_METRIC, 0.0)
        print(f"[promote] current Production: v{current.version} "
              f"(run {current.run_id}) PR-AUC={cur_prauc:.4f}")

        if current.run_id == best.info.run_id:
            print("[promote] best run is already in Production — nothing to do.")
            return False

        if only_if_better and best_prauc <= cur_prauc + config.PROMOTION_MIN_IMPROVEMENT:
            print(f"[promote] challenger ({best_prauc:.4f}) does not beat Production "
                  f"({cur_prauc:.4f}) by >= {config.PROMOTION_MIN_IMPROVEMENT} — skipping.")
            return False
    else:
        print("[promote] no model currently in Production.")

    # Register the best run's model as a new version, then promote it.
    version = register_and_promote(client, best.info.run_id, best_prauc)
    print(f"[promote] PROMOTED {config.REGISTERED_MODEL_NAME} v{version} "
          f"-> {config.PRODUCTION_STAGE} (PR-AUC={best_prauc:.4f})")
    return True


if __name__ == "__main__":
    promote(only_if_better="--only-if-better" in sys.argv)
