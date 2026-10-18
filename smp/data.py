"""Data loading and feature construction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["Open", "High", "Low", "Close", "Volume", "Average"]
CLOSE = FEATURES.index("Close")
TARGETS = ("price", "return")


def inside(path: str | Path, base: str | Path) -> Path:
    """Resolve path and refuse it unless it stays under base."""
    root = os.path.realpath(base)
    real = os.path.realpath(path)
    if real != root and not real.startswith(root + os.sep):
        raise ValueError(f"{path} resolves outside {root}")
    return Path(real)


def load_ohlcv(
    ticker: str, start: str, end: str, cache_dir: Path | None = None
) -> pd.DataFrame:
    """Daily OHLCV bars from Yahoo Finance, cached as CSV so reruns are offline and identical."""
    cache = None
    if cache_dir is not None:
        cache = inside(Path(cache_dir) / f"{ticker}_{start}_{end}.csv", cache_dir)
        if cache.exists():
            return pd.read_csv(cache, index_col=0, parse_dates=True)

    import yfinance as yf

    df = yf.download(
        ticker, start=start, end=end, multi_level_index=False, progress=False
    )
    if df.empty:
        raise RuntimeError(
            f"Yahoo Finance returned no rows for {ticker} {start}..{end}"
        )
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
    return df


@dataclass
class Series:
    """One ticker's model inputs, aligned so row t is trading day t.

    features[t] are the 6 inputs observed at the end of day t, close[t] is
    day t's close and prev_close[t] is the close of day t - 1. The first raw
    row is dropped in both target modes so they score exactly the same days.
    """

    ticker: str
    dates: pd.DatetimeIndex
    features: np.ndarray
    close: np.ndarray
    prev_close: np.ndarray


def build_features(df: pd.DataFrame, target: str, ticker: str = "") -> Series:
    """Turn raw OHLCV into the feature matrix for a target mode.

    price:  raw Open, High, Low, Close, Volume, Average (scaled later).
    return: log ratios against the previous close (log volume change for
            Volume), so the series is close to stationary and column 3 is
            the next-day log return the model is asked to predict.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    raw = df[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=float)
    average = (raw[:, 1] + raw[:, 2]) / 2
    raw = np.column_stack([raw, average])

    close = raw[1:, CLOSE]
    prev_close = raw[:-1, CLOSE]
    if target == "price":
        feats = raw[1:]
    else:
        prices = raw[1:, [0, 1, 2, 3, 5]]
        price_logs = np.log(prices / prev_close[:, None])
        volume = np.maximum(raw[:, 4], 1.0)
        vol_log = np.log(volume[1:] / volume[:-1])
        feats = np.column_stack([price_logs[:, :4], vol_log, price_logs[:, 4]])

    return Series(
        ticker=ticker,
        dates=pd.DatetimeIndex(df.index[1:]),
        features=feats,
        close=close,
        prev_close=prev_close,
    )
