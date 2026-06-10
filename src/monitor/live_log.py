"""Append-only log of live predictions, shared between the API (writer) and
the monitor (reader) via the mounted ``data/`` volume.

Kept dependency-light (pandas only) so the inference API can import it without
pulling in Evidently.
"""
from __future__ import annotations

import time

import pandas as pd

from src import config

_COLUMNS = config.FEATURE_COLS + ["prediction", "ts"]


def append_prediction(features: dict, proba: float) -> None:
    """Append one engineered feature row + its predicted probability."""
    row = {col: features.get(col) for col in config.FEATURE_COLS}
    row["prediction"] = proba
    row["ts"] = time.time()
    write_header = not config.LIVE_LOG_CSV.exists()
    pd.DataFrame([row], columns=_COLUMNS).to_csv(
        config.LIVE_LOG_CSV, mode="a", header=write_header, index=False
    )


def load_live_window(n: int = config.LIVE_WINDOW_SIZE) -> pd.DataFrame:
    """Return the most recent ``n`` logged predictions (empty frame if none)."""
    if not config.LIVE_LOG_CSV.exists():
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.read_csv(config.LIVE_LOG_CSV)
    return df.tail(n).reset_index(drop=True)


def reset_live_log() -> None:
    """Delete the live log so a fresh simulation starts clean."""
    config.LIVE_LOG_CSV.unlink(missing_ok=True)


def count() -> int:
    if not config.LIVE_LOG_CSV.exists():
        return 0
    return sum(1 for _ in config.LIVE_LOG_CSV.open()) - 1
