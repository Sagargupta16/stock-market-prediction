# CLAUDE.md

> This file stacks on top of the workspace root at `C:\Code\GitHub\`:
> - Root [`CLAUDE.md`](../../CLAUDE.md) -- voice, rules, routing map, references, skills, slash commands, conventions.
> - Root [`MEMORY.md`](../../MEMORY.md) -- live facts across repos.
> - Root [`STATUS.md`](../../STATUS.md) -- live PR/CI/security dashboard.
> - [`.claude/resources/`](../../.claude/resources/README.md) -- deep reference for collaboration, workflow, git, OSS, debugging, voice.
>
> Read those first. The guidance below only adds **repo-specific context** -- it does not override anything in the root.

## Project

LSTM stock-price predictor for AAPL/MSFT/AMZN/TSLA. Two surfaces: `SMP.ipynb` (one-window Colab demo, badge in README) and the `smp/` package, a walk-forward benchmark of naive vs LSTM (price target) vs LSTM (log-return target) vs zero-shot Chronos-2, with permutation importance. Educational/evaluation project, not a trading signal.

Public repo: [Sagargupta16/stock-market-prediction](https://github.com/Sagargupta16/stock-market-prediction). No deployed service.

## Stack

- **Language**: Python 3.14 (pinned in `.python-version`)
- **Framework**: PyTorch (nn.LSTM), scikit-learn (MinMaxScaler, StandardScaler, MSE), yfinance, pandas, matplotlib, chronos-forecasting (Chronos-2 baseline), pytest
- **Database**: none -- data fetched live from Yahoo Finance
- **Package manager**: pip via `requirements.txt`, a uv-generated hash lock of `requirements.in` (Renovate monthly through its pip-compile manager)
- **Deploy target**: none -- local Jupyter or Google Colab

## Run

```
pip install -r requirements.txt
jupyter notebook SMP.ipynb
```

Run cells top to bottom: fetch/preprocess -> model definition -> training -> test split -> evaluation -> moving averages -> permutation importance.

Benchmark (writes `results/benchmark.{json,md}` and `docs/*.png`, caches Yahoo CSVs in `data/cache/`, gitignored):

```
python -m smp.benchmark                # 8 folds x 30 days, 150 epochs, Chronos-2 (downloads amazon/chronos-2 from Hugging Face)
python -m smp.benchmark --no-chronos   # offline after the first data fetch
python -m smp.benchmark --end today    # latest data instead of 2020-2022
```

## Test

```
python -m pytest -q
```

`tests/test_pipeline.py` is offline (synthetic random walks): feature alignment, fold boundaries, scaler-ignores-test-rows, windows never see the label row, inverse-scale round trip, metrics, permutation importance, seeded training, and an end-to-end `run_benchmark`. CI runs it after the notebook compile step.

## Entry points

- `SMP.ipynb` -- one-window demo: data fetch, LSTM definition, training, evaluation, plots
- `python -m smp.benchmark` -- walk-forward benchmark CLI (`smp/benchmark.py:main`)

## Key files

- `SMP.ipynb` -- self-contained so Colab can open it without the repo; it does NOT import `smp/`, so pipeline changes go in both places
- `smp/data.py` -- yfinance load with CSV cache, `build_features` (price or log-return features; drops the first row in both modes so they score the same days)
- `smp/evaluate.py` -- `fold_starts` (expanding, contiguous folds), leak-free `prepare_fold`, `to_price`, RMSE/MAPE/direction metrics
- `smp/model.py` -- batch_first LSTM, seeded mini-batch `train_lstm`, CUDA if available
- `smp/importance.py` -- permutation importance; `smp/chronos_baseline.py` -- Chronos-2 one-step median
- `results/benchmark.md` -- the committed measured tables; README numbers are copied from here, never typed by hand
- `requirements.in` -- direct pins (torch, yfinance, pandas, scikit-learn, matplotlib, numpy, jupyter); bump here, then rerun the command in `requirements.txt`'s header
- `requirements.txt` -- generated `--generate-hashes` lock; CI installs it with `--require-hashes`, so never hand-edit a version in it

## Gotchas

- Needs network at runtime: `yf.download()` pulls 2020-01-01 to 2023-01-01 data live; nothing is cached in the repo.
- One shared model is trained on windows from all 4 stocks, but each stock has its OWN MinMaxScaler -- always inverse-transform with the matching scaler.
- Hold-out: `N = 30` (cell 1) is excluded from each stock's training windows AND from its scaler fit (`fetch_preprocess(..., test_days=N)`). Training windows are built per ticker; never build them on `data_combined` again (that leaked the test set and spanned tickers). `data_combined` only feeds the moving-average cell.
- AMZN's test-window closes sit below its training min, so scaled test values can be slightly negative. That is expected, not a bug.
- `evaluate_model()` prints a naive previous-close baseline MSE next to the model MSE; report both.
- The last cell computes permutation importance on the held-out windows (10 shuffles per feature, seeded). It replaced a hardcoded placeholder chart in 1.4.0.
- Training is mini-batched (batch 64, `torch.manual_seed(0)`, CUDA if present); the per-window loop it replaced took about 70 min on CPU.
- GitHub repo name is lowercase `stock-market-prediction`; README and notebook Colab links both use it.
- `create_inout_sequences` is defined twice (training and test cells); keep them in sync if editing.
- Return-target LSTM predicts the standardized next-day log return; `to_price` maps it back with `prev_close * exp(r)`. Naive has no direction call (it predicts no change), so its direction cell is n/a and "Always up" is the direction baseline.
- Chronos-2 is zero-shot: its pretraining data may include public prices from the test period, so treat its row as optimistic.
- `fold_starts` needs at least 250 training windows in the first fold; 8 folds x 30 days fits 2020-2022, more folds need a longer `--start`.
- CI exports the notebook with `jupyter nbconvert --to python` (not `--to script`: the kernelspec has no language_info, so `script` writes `SMP.txt`) and runs `py_compile` on it.

## Dataset

- Location: fetched at runtime from Yahoo Finance via yfinance (AAPL, MSFT, AMZN, TSLA)
- Size: ~750 trading days per ticker (2020-01-01 to 2023-01-01)
- Format: in-memory pandas DataFrame -> scaled torch.FloatTensor; 6 features (Open, High, Low, Close, Volume, Average)

## Training

- Command: run the training cell in `SMP.ipynb` (150 epochs, batch 64, Adam lr=0.001, MSELoss, seq_length=5, hidden=100), or `python -m smp.benchmark` (same settings, one model per fold and target)
- Approx time: see the README Limitations and `results/benchmark.md` Run time table (both measured 2026-10-18)
- Checkpoints: none -- model is never saved to disk

## Eval metric

- Notebook: MSE on the held-out last 30 days per stock, printed by `evaluate_model()` alongside the naive previous-close baseline; plots actual vs predicted
- Benchmark: RMSE ($), MAPE (%), direction accuracy (%) per stock, mean and std over 8 folds, plus how many stock-folds each model beats naive on

## Artifacts

- Model output: none persisted -- model exists only in the live notebook session
- Versioning: n/a
