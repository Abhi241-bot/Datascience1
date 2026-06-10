"""Build drift-inclusive datasets for the closed loop.

Models the realistic situation after a drift event: newly-collected, *labeled*
transactions from the shifted regime become available. Appending shifted copies
of labeled rows lets the retrain learn the new regime, and lets the promotion
gate score the incumbent and challenger on the *same* current distribution.

The shift here matches the live simulator (``config.DRIFT_FEATURE_SHIFT`` /
``DRIFT_AMOUNT_MULTIPLIER``), so the labeled data describes the same drift the
monitor detects.
"""
from __future__ import annotations

import pandas as pd

from src import config


def apply_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the deliberate feature shift applied.

    Operates on the raw feature columns (V1..V28 shifted, Amount scaled) and
    preserves the label column.
    """
    shifted = df.copy()
    shifted[config.PCA_COLS] = shifted[config.PCA_COLS] + config.DRIFT_FEATURE_SHIFT
    shifted[config.AMOUNT_COL] = shifted[config.AMOUNT_COL] * config.DRIFT_AMOUNT_MULTIPLIER
    return shifted


def augment_with_drift(df: pd.DataFrame, frac: float = 1.0) -> pd.DataFrame:
    """Original rows + shifted copies of a ``frac`` sample of them (shuffled)."""
    sample = df.sample(frac=frac, random_state=config.RANDOM_SEED)
    combined = pd.concat([df, apply_drift(sample)], ignore_index=True)
    return combined.sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)
