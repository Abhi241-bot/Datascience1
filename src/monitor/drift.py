"""Drift computation with Evidently AI.

Compares the reference window (training distribution) against the most recent
live predictions and reports dataset-level data drift. The summary is the
signal the closed loop (Phase 5) acts on.
"""
from __future__ import annotations

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src import config
from src.monitor import live_log

_COLUMN_MAPPING = ColumnMapping(numerical_features=config.FEATURE_COLS)


def load_reference() -> pd.DataFrame:
    df = pd.read_csv(config.REFERENCE_CSV)
    return df[config.FEATURE_COLS]


def _extract_summary(report: Report) -> dict:
    """Pull the dataset-drift numbers out of the Evidently report dict."""
    for metric in report.as_dict().get("metrics", []):
        result = metric.get("result", {})
        if "share_of_drifted_columns" in result:
            share = float(result["share_of_drifted_columns"])
            return {
                "dataset_drift": bool(result.get("dataset_drift", False)),
                "n_drifted_columns": int(result.get("number_of_drifted_columns", 0)),
                "n_columns": int(result.get("number_of_columns", len(config.FEATURE_COLS))),
                "share_drifted_columns": share,
                "drift_detected": share >= config.DRIFT_SHARE_THRESHOLD,
                "threshold": config.DRIFT_SHARE_THRESHOLD,
            }
    return {"drift_detected": False, "share_drifted_columns": 0.0,
            "n_drifted_columns": 0, "n_columns": len(config.FEATURE_COLS),
            "dataset_drift": False, "threshold": config.DRIFT_SHARE_THRESHOLD}


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[Report, dict]:
    """Run Evidently on two feature frames and return (report, summary)."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current,
               column_mapping=_COLUMN_MAPPING)
    summary = _extract_summary(report)
    summary["n_live_rows"] = int(len(current))
    return report, summary


def build_report(current: pd.DataFrame | None = None) -> tuple[Report, dict]:
    """Build a drift report comparing reference vs the current live window.

    Returns (report, summary). If there is no live data yet, returns a report
    comparing the reference to itself (no drift) so the dashboard still renders.
    """
    reference = load_reference()
    if current is None:
        live = live_log.load_live_window()
        current = live[config.FEATURE_COLS] if len(live) else reference.copy()
    return run_drift_report(reference, current)


def compute_summary() -> dict:
    """Convenience: just the drift summary (used by the retraining trigger)."""
    _, summary = build_report()
    return summary


def save_report(report: Report, path=None) -> str:
    path = str(path or config.DRIFT_REPORT_HTML)
    report.save_html(path)
    return path
