"""Download the credit-card fraud dataset from a public mirror.

Idempotent: skips the download if a valid file already exists.
Run with:  python -m src.data.download
"""
from __future__ import annotations

import sys

import requests

from src import config


def download(force: bool = False) -> None:
    dest = config.RAW_CSV
    if dest.exists() and not force:
        n_lines = sum(1 for _ in dest.open())
        print(f"[download] {dest} already exists ({n_lines - 1} rows) — skipping.")
        return

    print(f"[download] fetching {config.DATASET_URL}")
    with requests.get(config.DATASET_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MiB
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(f"\r[download] {downloaded >> 20} / {total >> 20} MiB "
                          f"({pct:5.1f}%)", end="")
        print()

    n_rows = sum(1 for _ in dest.open()) - 1  # minus header
    print(f"[download] saved {dest} ({n_rows} rows)")
    if n_rows != config.EXPECTED_ROWS:
        print(f"[download] WARNING: expected {config.EXPECTED_ROWS} rows, "
              f"got {n_rows}", file=sys.stderr)


if __name__ == "__main__":
    download(force="--force" in sys.argv)
