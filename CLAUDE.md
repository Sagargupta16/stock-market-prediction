# CLAUDE.md

> This file stacks on top of the workspace root at `C:\Code\GitHub\`:
> - Root [`CLAUDE.md`](../../CLAUDE.md) -- voice, rules, routing map, references, skills, slash commands, conventions.
> - Root [`MEMORY.md`](../../MEMORY.md) -- live facts across repos.
> - Root [`STATUS.md`](../../STATUS.md) -- live PR/CI/security dashboard.
> - [`.claude/resources/`](../../.claude/resources/README.md) -- deep reference for collaboration, workflow, git, OSS, debugging, voice.
>
> Read those first. The guidance below only adds **repo-specific context** -- it does not override anything in the root.

## Project

LSTM stock-price predictor for AAPL/MSFT/AMZN/TSLA, built as a single Jupyter notebook. Educational/demo project, runnable in Colab (badge in README).

Public repo: [Sagargupta16/stock-market-prediction](https://github.com/Sagargupta16/stock-market-prediction). No deployed service.

## Stack

- **Language**: Python 3.14 (pinned in `.python-version`)
- **Framework**: PyTorch (nn.LSTM), scikit-learn (MinMaxScaler, MSE), yfinance, pandas, matplotlib
- **Database**: none -- data fetched live from Yahoo Finance
- **Package manager**: pip via `requirements.txt` (pinned versions, Renovate monthly)
- **Deploy target**: none -- local Jupyter or Google Colab

## Run

```
pip install -r requirements.txt
jupyter notebook SMP.ipynb
```

Run cells top to bottom: fetch/preprocess -> model definition -> training -> test split -> evaluation -> moving averages -> feature importance.

## Test

No test suite. Verification = notebook runs end to end and prints per-stock MSE.

## Entry points

- `SMP.ipynb` -- the entire project: data fetch, LSTM definition, training, evaluation, plots

## Key files

- `SMP.ipynb` -- all code lives here; there are no `.py` modules
- `requirements.txt` -- pinned deps (torch, yfinance, pandas, scikit-learn, matplotlib, numpy, jupyter)

## Gotchas

- Needs network at runtime: `yf.download()` pulls 2020-01-01 to 2023-01-01 data live; nothing is cached in the repo.
- One shared model is trained on windows from all 4 stocks, but each stock has its OWN MinMaxScaler -- always inverse-transform with the matching scaler.
- Hold-out: `N = 30` (cell 1) is excluded from each stock's training windows AND from its scaler fit (`fetch_preprocess(..., test_days=N)`). Training windows are built per ticker; never build them on `data_combined` again (that leaked the test set and spanned tickers). `data_combined` only feeds the moving-average cell.
- AMZN's test-window closes sit below its training min, so scaled test values can be slightly negative. That is expected, not a bug.
- `evaluate_model()` prints a naive previous-close baseline MSE next to the model MSE; report both.
- The last cell's "Illustrative Feature Weights" chart uses a hardcoded placeholder dict, not values computed from the model. Don't present it as a real result.
- Training loops per-sequence with hidden-state resets (no batching); 150 epochs is slow on CPU (tens of minutes).
- GitHub repo name is lowercase `stock-market-prediction`; README and notebook Colab links both use it.
- `create_inout_sequences` is defined twice (training and test cells); keep them in sync if editing.
- CI exports the notebook with `jupyter nbconvert --to python` (not `--to script`: the kernelspec has no language_info, so `script` writes `SMP.txt`) and runs `py_compile` on it.

## Dataset

- Location: fetched at runtime from Yahoo Finance via yfinance (AAPL, MSFT, AMZN, TSLA)
- Size: ~750 trading days per ticker (2020-01-01 to 2023-01-01)
- Format: in-memory pandas DataFrame -> scaled torch.FloatTensor; 6 features (Open, High, Low, Close, Volume, Average)

## Training

- Command: run the training cell in `SMP.ipynb` (150 epochs, Adam lr=0.001, MSELoss, seq_length=5, hidden=100)
- Approx time: about 70 min on a laptop CPU for 150 epochs (measured 2026-10-18 via `nbconvert --execute`, per-sequence loop, no batching). Pass `--ExecutePreprocessor.timeout=7200` or the training cell times out at 3600s.
- Checkpoints: none -- model is never saved to disk

## Eval metric

- MSE on the held-out last 30 days per stock, printed by `evaluate_model()` alongside the naive previous-close baseline; plots actual vs predicted

## Artifacts

- Model output: none persisted -- model exists only in the live notebook session
- Versioning: n/a
