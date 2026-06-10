"""Simulate live traffic against the API, then inject a drift event.

Replays held-out transactions through ``POST /predict``. The first half is
genuine data (monitor shows "no drift"); the second half is shifted so feature
distributions move (monitor flips to "drift detected"). This is the live demo.

Run with:
    python -m src.simulate.stream --n 1500 --drift-after 750 --delay 0.0
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import requests

from src import config

# Features perturbed during the drift phase (shared with the retrain builder so
# the simulated traffic and the labeled retrain data describe the same shift).
_DRIFT_SHIFT = config.DRIFT_FEATURE_SHIFT
_AMOUNT_MULTIPLIER = config.DRIFT_AMOUNT_MULTIPLIER


def _load_rows() -> pd.DataFrame:
    # Replay in-distribution traffic: the reference window is the training
    # distribution, so genuine live traffic drawn from it reads as "no drift"
    # until we deliberately shift it. (A time-ordered test split carries its
    # own covariate shift, which would muddy the demo.)
    df = pd.read_csv(config.TRAIN_CSV)
    return df[config.FEATURE_COLS].reset_index(drop=True)


def _make_payload(row: pd.Series, drift: bool) -> dict:
    payload = {col: float(row[col]) for col in config.FEATURE_COLS}
    if drift:
        for col in config.PCA_COLS:
            payload[col] += _DRIFT_SHIFT
        payload[config.AMOUNT_COL] = float(row[config.AMOUNT_COL]) * _AMOUNT_MULTIPLIER
    return payload


def _summary(monitor_url: str) -> dict:
    return requests.get(f"{monitor_url}/drift-summary", timeout=60).json()


def _print_summary(label: str, s: dict) -> None:
    flag = "DRIFT DETECTED" if s.get("drift_detected") else "no drift"
    print(f"[stream] {label}: {flag}  "
          f"(drifted {s.get('n_drifted_columns')}/{s.get('n_columns')}, "
          f"share={s.get('share_drifted_columns'):.2f}, live_rows={s.get('n_live_rows')})")


def run(n: int, drift_after: int, delay: float,
        api_url: str, monitor_url: str, seed: int = config.RANDOM_SEED) -> None:
    rows = _load_rows().sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    predict_url = f"{api_url}/predict"

    print(f"[stream] resetting monitor live log at {monitor_url}")
    requests.post(f"{monitor_url}/reset", timeout=60)

    print(f"[stream] sending {n} transactions; injecting drift after {drift_after}")
    fraud_flags = 0
    for i in range(n):
        drift = i >= drift_after
        resp = requests.post(predict_url, json=_make_payload(rows.iloc[i], drift), timeout=60)
        resp.raise_for_status()
        fraud_flags += int(resp.json()["label"])

        if (i + 1) == drift_after:
            print(f"[stream] --- sent {drift_after} genuine transactions ---")
            _print_summary("after genuine traffic", _summary(monitor_url))
            print("[stream] >>> INJECTING DRIFT (shifting feature distributions) <<<")
        elif (i + 1) % 250 == 0:
            print(f"[stream] {i + 1}/{n} sent ({'drifted' if drift else 'genuine'})")

        if delay:
            time.sleep(delay)

    print(f"[stream] --- done: {n} sent, {fraud_flags} flagged fraud ---")
    _print_summary("after drift injection", _summary(monitor_url))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--drift-after", type=int, default=750)
    p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("--api-url", default=config.API_URL)
    p.add_argument("--monitor-url", default=config.MONITOR_URL)
    args = p.parse_args()
    run(args.n, args.drift_after, args.delay, args.api_url, args.monitor_url)
