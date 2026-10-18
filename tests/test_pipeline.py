"""Offline tests: synthetic prices only, no network."""

import numpy as np
import pandas as pd
import pytest

from smp.benchmark import Config, render_markdown, run_benchmark
from smp.data import CLOSE, build_features
from smp.evaluate import (
    direction_accuracy,
    fit_scaler,
    fold_starts,
    inverse_close,
    make_windows,
    mape,
    prepare_fold,
    rmse,
    to_price,
)
from smp.importance import permutation_importance
from smp.model import predict, train_lstm


def random_walk(n=600, seed=0, start=100.0):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = close * np.exp(rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def test_build_features_aligns_close_and_prev_close():
    df = random_walk(50)
    for target in ("price", "return"):
        s = build_features(df, target)
        assert s.features.shape == (49, 6)
        np.testing.assert_allclose(s.close, df["Close"].to_numpy()[1:])
        np.testing.assert_allclose(s.prev_close, df["Close"].to_numpy()[:-1])
    ret = build_features(df, "return")
    np.testing.assert_allclose(
        ret.features[:, CLOSE], np.log(ret.close / ret.prev_close)
    )


def test_build_features_rejects_unknown_target():
    with pytest.raises(ValueError):
        build_features(random_walk(20), "level")


def test_fold_starts_are_contiguous_and_end_at_last_row():
    starts = fold_starts(500, 4, 30, 5)
    assert starts == [380, 410, 440, 470]
    assert starts[-1] + 30 == 500
    with pytest.raises(ValueError):
        fold_starts(300, 8, 30, 5)


def test_scaler_ignores_test_rows():
    feats = np.ones((100, 6))
    feats[90:, :] = 1000.0  # spike only inside the test block
    scaler = fit_scaler(feats, 90, "price")
    assert scaler.data_max_.max() == 1.0


def test_windows_never_use_the_label_row_or_later():
    scaled = np.arange(40, dtype=float)[:, None].repeat(6, axis=1)
    x, y = make_windows(scaled, 10, 20, 5)
    assert x.shape == (10, 5, 6)
    # Every input row index is strictly less than its label row index
    assert np.all(x[:, :, CLOSE].max(axis=1) < y)
    np.testing.assert_array_equal(y, np.arange(10, 20))


def test_train_and_test_windows_do_not_overlap():
    s = build_features(random_walk(400), "price")
    fold = prepare_fold(s, 300, 30, 5, "price")
    assert len(fold.x_train) == 300 - 5
    assert len(fold.x_test) == 30
    np.testing.assert_array_equal(fold.test_rows, np.arange(300, 330))
    # Last training label is row 299, the day before the first test day
    scaled = fold.scaler.transform(s.features)
    assert fold.y_train[-1] == scaled[299, CLOSE]


@pytest.mark.parametrize("target", ["price", "return"])
def test_inverse_scaling_round_trips_to_actual_close(target):
    s = build_features(random_walk(300), target)
    fold = prepare_fold(s, 250, 30, 5, target)
    back = to_price(fold.y_test, fold.scaler, s, fold.test_rows, target)
    np.testing.assert_allclose(back, s.close[fold.test_rows], rtol=1e-9)
    raw = inverse_close(fold.scaler, fold.y_test, 6)
    assert raw.shape == (30,)


def test_metrics():
    actual = np.array([100.0, 110.0])
    pred = np.array([90.0, 110.0])
    prev = np.array([95.0, 105.0])
    assert rmse(actual, pred) == pytest.approx(np.sqrt(50))
    assert mape(actual, pred) == pytest.approx(5.0)
    # day 1: actual up, pred down -> wrong; day 2: both up -> right
    assert direction_accuracy(actual, pred, prev) == 50.0


def test_permutation_importance_is_zero_for_unused_features():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5, 6))
    y = x[:, -1, CLOSE]
    imp = permutation_importance(lambda a: a[:, -1, CLOSE], x, y, repeats=3)
    assert imp[CLOSE] > 1.0
    np.testing.assert_array_equal(np.delete(imp, CLOSE), 0.0)


def test_training_is_seeded_and_learns():
    rng = np.random.default_rng(1)
    x = rng.uniform(size=(256, 5, 6)).astype(np.float32)
    y = x[:, -1, CLOSE]
    a = train_lstm(x, y, epochs=60, lr=1e-2, hidden_size=16, seed=7)
    b = train_lstm(x, y, epochs=60, lr=1e-2, hidden_size=16, seed=7)
    pa, pb = predict(a, x), predict(b, x)
    np.testing.assert_allclose(pa, pb)
    assert np.mean((pa - y) ** 2) < 0.5 * np.var(y)


def test_benchmark_end_to_end_offline():
    frames = {t: random_walk(420, seed=i) for i, t in enumerate(["AAA", "BBB"])}
    cfg = Config(
        tickers=list(frames), folds=2, test_days=20, epochs=1, importance_repeats=1
    )
    result = run_benchmark(frames, cfg, chronos=None, log=lambda *_: None)
    rows = pd.DataFrame(result["rows"])
    assert set(rows["model"]) == {"Naive", "LSTM (price)", "LSTM (returns)"}
    assert len(rows) == 3 * 2 * 2
    # Both targets score the same test days
    days = rows.groupby("model")[["test_from", "test_to"]].apply(
        lambda d: tuple(map(tuple, d.to_numpy()))
    )
    assert days.nunique() == 1
    assert rows[rows["model"] == "Naive"]["direction"].isna().all()
    md = render_markdown(result)
    assert "Head to head against naive" in md
    assert "of 4 |" in md
