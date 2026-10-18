"""Permutation feature importance on held-out windows."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def permutation_importance(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    *,
    repeats: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Mean rise in test MSE when one feature is shuffled across windows.

    Each feature's whole 5-day slice moves together to another window, which
    breaks its link to the label while keeping its marginal distribution.
    Returns one value per feature in the units of y squared.
    """
    rng = np.random.default_rng(seed)
    base = float(np.mean((predict_fn(x) - y) ** 2))
    out = np.zeros(x.shape[2])
    for f in range(x.shape[2]):
        rises = []
        for _ in range(repeats):
            shuffled = x.copy()
            shuffled[:, :, f] = x[rng.permutation(len(x)), :, f]
            rises.append(float(np.mean((predict_fn(shuffled) - y) ** 2)) - base)
        out[f] = np.mean(rises)
    return out
