"""Drift detection tests: the detector must stay quiet on in-distribution data
and fire when the distribution is deliberately shifted."""
import numpy as np
import pandas as pd

from src import config
from src.data.drifted import apply_drift
from src.monitor import drift


def _frame(n: int, shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(loc=shift, size=n) for col in config.PCA_COLS}
    data[config.AMOUNT_COL] = np.log1p(rng.uniform(0, 1000, size=n))
    return pd.DataFrame(data)[config.FEATURE_COLS]


def test_no_drift_for_same_distribution():
    reference = _frame(800, seed=1)
    current = _frame(800, seed=2)  # same distribution, different sample
    _, summary = drift.run_drift_report(reference, current)
    assert summary["drift_detected"] is False
    assert summary["share_drifted_columns"] < config.DRIFT_SHARE_THRESHOLD


def test_drift_detected_for_shifted_distribution():
    reference = _frame(800, seed=1)
    current = _frame(800, shift=5.0, seed=2)  # large mean shift on every feature
    _, summary = drift.run_drift_report(reference, current)
    assert summary["drift_detected"] is True
    assert summary["share_drifted_columns"] >= config.DRIFT_SHARE_THRESHOLD


def test_apply_drift_shifts_features():
    df = _frame(50)
    shifted = apply_drift(df.assign(**{config.TARGET_COL: 0}))
    assert np.allclose(
        shifted[config.PCA_COLS].values,
        df[config.PCA_COLS].values + config.DRIFT_FEATURE_SHIFT,
    )
    assert np.allclose(
        shifted[config.AMOUNT_COL].values,
        df[config.AMOUNT_COL].values * config.DRIFT_AMOUNT_MULTIPLIER,
    )
