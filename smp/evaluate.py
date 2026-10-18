"""Rolling-origin folds, leak-free scaling, windowing and metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .data import CLOSE, Series


def fold_starts(
    n_rows: int, n_folds: int, test_days: int, seq_len: int, min_train: int = 250
) -> list[int]:
    """Row index where each fold's test block begins; the last block ends at the last row.

    Folds are contiguous, non-overlapping and expanding: fold k trains on every
    row before its test block and never on anything after it.
    """
    starts = [n_rows - (n_folds - k) * test_days for k in range(n_folds)]
    if starts[0] - seq_len < min_train:
        raise ValueError(
            f"{n_folds} folds of {test_days} days leave {starts[0] - seq_len} training windows "
            f"in the first fold; need at least {min_train}. Use fewer folds or a longer date range."
        )
    return starts


def fit_scaler(
    features: np.ndarray, test_start: int, target: str
) -> MinMaxScaler | StandardScaler:
    """Fit on rows before the test block only, so test min/max/mean never reach training."""
    scaler = MinMaxScaler() if target == "price" else StandardScaler()
    return scaler.fit(features[:test_start])


def make_windows(
    scaled: np.ndarray, start: int, stop: int, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Windows whose label row j is in [start, stop): inputs rows j-seq_len..j-1, label row j's Close column."""
    start = max(start, seq_len)
    idx = np.arange(start, stop)
    x = np.stack([scaled[j - seq_len : j] for j in idx])
    y = scaled[idx, CLOSE]
    return x, y


def inverse_close(
    scaler: MinMaxScaler | StandardScaler, values: np.ndarray, n_features: int
) -> np.ndarray:
    """Undo scaling for the Close column only."""
    dummy = np.zeros((len(values), n_features))
    dummy[:, CLOSE] = values
    return scaler.inverse_transform(dummy)[:, CLOSE]


def to_price(
    pred_scaled: np.ndarray, scaler, series: Series, rows: np.ndarray, target: str
) -> np.ndarray:
    """Model output for rows -> predicted close in dollars."""
    value = inverse_close(scaler, pred_scaled, series.features.shape[1])
    if target == "price":
        return value
    return series.prev_close[rows] * np.exp(value)


@dataclass
class Fold:
    """Scaled train and test windows for one ticker and one fold."""

    scaler: MinMaxScaler | StandardScaler
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_rows: np.ndarray


def prepare_fold(
    series: Series, test_start: int, test_days: int, seq_len: int, target: str
) -> Fold:
    scaler = fit_scaler(series.features, test_start, target)
    scaled = scaler.transform(series.features)
    x_train, y_train = make_windows(scaled, seq_len, test_start, seq_len)
    x_test, y_test = make_windows(scaled, test_start, test_start + test_days, seq_len)
    return Fold(
        scaler,
        x_train,
        y_train,
        x_test,
        y_test,
        np.arange(test_start, test_start + test_days),
    )


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual) / np.abs(actual)) * 100)


def direction_accuracy(actual: np.ndarray, pred: np.ndarray, prev: np.ndarray) -> float:
    """Share of days where the forecast gets up vs not-up right, in percent."""
    return float(np.mean((pred > prev) == (actual > prev)) * 100)


def score(
    actual: np.ndarray, pred: np.ndarray, prev: np.ndarray, with_direction: bool = True
) -> dict[str, float | None]:
    """direction is None for forecasts that never call a move (naive)."""
    return {
        "rmse": rmse(actual, pred),
        "mape": mape(actual, pred),
        "direction": direction_accuracy(actual, pred, prev) if with_direction else None,
    }
