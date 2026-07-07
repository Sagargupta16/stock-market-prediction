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
- One shared model is trained on the concatenated 4-stock data, but each stock has its OWN MinMaxScaler -- always inverse-transform with the matching scaler.
- The "Feature Importances" chart (last cell) uses a hardcoded dummy dict, not values computed from the model. Don't present it as a real result.
- Training loops per-sequence with hidden-state resets (no batching); 150 epochs is slow on CPU. README's "batch training" claim does not match the code.
- GitHub repo name is lowercase `stock-market-prediction` (matches the local folder), but the Colab badge in README and notebook still links to the old `Stock-market-prediction` casing -- GitHub redirects, so both resolve.
- `create_inout_sequences` is defined twice (training and test cells); keep them in sync if editing.

## Dataset

- Location: fetched at runtime from Yahoo Finance via yfinance (AAPL, MSFT, AMZN, TSLA)
- Size: ~750 trading days per ticker (2020-01-01 to 2023-01-01)
- Format: in-memory pandas DataFrame -> scaled torch.FloatTensor; 6 features (Open, High, Low, Close, Volume, Average)

## Training

- Command: run the training cell in `SMP.ipynb` (150 epochs, Adam lr=0.001, MSELoss, seq_length=5, hidden=100)
- Approx time: minutes on CPU (per-sequence loop, no batching)
- Checkpoints: none -- model is never saved to disk

## Eval metric

- MSE on the last 30 days per stock, printed by `evaluate_model()`; no fixed target, plots actual vs predicted

## Artifacts

- Model output: none persisted -- model exists only in the live notebook session
- Versioning: n/a
