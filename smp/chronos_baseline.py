"""Zero-shot Chronos-2 one-step-ahead forecasts."""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "amazon/chronos-2"


def load_pipeline(model_id: str = DEFAULT_MODEL, device: str = "cpu"):
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(model_id, device_map=device)


def forecast_next_close(
    pipeline, contexts: list[np.ndarray], batch_size: int = 256
) -> np.ndarray:
    """Median next-day close for each context (all closes strictly before the target day)."""
    quantiles, _ = pipeline.predict_quantiles(
        [np.asarray(c, dtype=np.float32) for c in contexts],
        prediction_length=1,
        quantile_levels=[0.5],
        batch_size=batch_size,
    )
    return np.array([float(q.reshape(-1)[0]) for q in quantiles])
