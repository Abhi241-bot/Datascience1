"""Unit tests for feature engineering and the time-ordered split."""
import numpy as np
import pandas as pd

from src import config
from src.data.preprocess import _time_ordered_split, engineer_features


def _raw_frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {col: rng.normal(size=n) for col in config.PCA_COLS}
    data[config.AMOUNT_COL] = rng.uniform(0, 1000, size=n)
    data[config.TIME_COL] = np.arange(n)[::-1]  # reversed, to test sorting
    data[config.TARGET_COL] = rng.integers(0, 2, size=n)
    return pd.DataFrame(data)


def test_engineer_features_columns_and_order():
    out = engineer_features(_raw_frame())
    assert list(out.columns) == config.FEATURE_COLS
    assert config.TIME_COL not in out.columns  # Time is dropped


def test_engineer_features_log_amount():
    df = _raw_frame()
    out = engineer_features(df)
    expected = np.log1p(df[config.AMOUNT_COL].astype("float64"))
    assert np.allclose(out[config.AMOUNT_COL].values, expected.values)


def test_engineer_features_passes_through_pca():
    df = _raw_frame()
    out = engineer_features(df)
    for col in config.PCA_COLS:
        assert np.allclose(out[col].values, df[col].values)


def test_time_ordered_split_is_chronological_and_partitioned():
    df = _raw_frame(1000)
    train, val, test = _time_ordered_split(df)
    # Sizes follow the configured fractions, and partitions are disjoint.
    assert len(train) == int(1000 * config.TRAIN_FRAC)
    assert len(train) + len(val) + len(test) == 1000
    # Every train timestamp precedes every val timestamp (chronological).
    assert train[config.TIME_COL].max() <= val[config.TIME_COL].min()
    assert val[config.TIME_COL].max() <= test[config.TIME_COL].min()
