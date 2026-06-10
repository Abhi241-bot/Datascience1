"""Clean, engineer features, build a time-ordered split, and save a reference
window for drift monitoring.

Run with:  python -m src.data.preprocess

Design note
-----------
`engineer_features` is a module-level function so it can be wrapped in a
scikit-learn ``FunctionTransformer`` and pickled inside the model pipeline.
That guarantees the exact same transformation runs at train time and at
serving time (train/serve parity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw transaction columns -> model feature matrix.

    Keeps the 28 PCA components as-is and replaces the heavily right-skewed
    ``Amount`` with ``log1p(Amount)``. ``Time`` is intentionally dropped (it is
    only used to order the split, not as a predictive feature).
    """
    out = pd.DataFrame(index=df.index)
    for col in config.PCA_COLS:
        out[col] = df[col].astype("float64")
    out[config.AMOUNT_COL] = np.log1p(df[config.AMOUNT_COL].astype("float64"))
    return out[config.FEATURE_COLS]


def _time_ordered_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows by ascending Time so val/test are 'in the future' of train."""
    df = df.sort_values(config.TIME_COL).reset_index(drop=True)
    n = len(df)
    n_train = int(n * config.TRAIN_FRAC)
    n_val = int(n * config.VAL_FRAC)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    return train, val, test


def preprocess() -> None:
    print(f"[preprocess] loading {config.RAW_CSV}")
    df = pd.read_csv(config.RAW_CSV)

    # Basic cleaning: drop exact duplicates, ensure target is int.
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df[config.TARGET_COL] = df[config.TARGET_COL].astype(int)
    print(f"[preprocess] dropped {before - len(df)} duplicate rows; "
          f"{len(df)} remain")

    train, val, test = _time_ordered_split(df)
    for name, part in (("train", train), ("val", val), ("test", test)):
        pos = int(part[config.TARGET_COL].sum())
        print(f"[preprocess] {name:5s}: {len(part):>7} rows, "
              f"{pos} fraud ({100 * pos / len(part):.3f}%)")

    train.to_csv(config.TRAIN_CSV, index=False)
    val.to_csv(config.VAL_CSV, index=False)
    test.to_csv(config.TEST_CSV, index=False)
    print(f"[preprocess] wrote splits to {config.DATA_PROCESSED}")

    # Reference window for drift = a sample of the engineered training features
    # plus the target. This is the baseline Evidently compares live data to.
    sample_n = min(config.REFERENCE_SAMPLE_SIZE, len(train))
    ref_raw = train.sample(n=sample_n, random_state=config.RANDOM_SEED)
    reference = engineer_features(ref_raw)
    reference[config.TARGET_COL] = ref_raw[config.TARGET_COL].values
    reference.to_csv(config.REFERENCE_CSV, index=False)
    print(f"[preprocess] wrote reference window ({sample_n} rows) to "
          f"{config.REFERENCE_CSV}")


if __name__ == "__main__":
    preprocess()
