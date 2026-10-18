# Changelog

## [1.4.0] - 2026-10-18

- Add the `smp/` package and `python -m smp.benchmark`: walk-forward evaluation over 8 expanding 30-day folds per stock, scoring naive, LSTM on price, LSTM on log returns and zero-shot Chronos-2 on the same days with RMSE, MAPE and direction accuracy (plus an always-up baseline)
- Commit the measured run in `results/benchmark.{json,md}` and its charts in `docs/`; `--from-json` redraws them without retraining, `--end today` scores current data
- Replace the notebook's placeholder feature chart with permutation importance computed on the held-out windows; the benchmark reports it for both LSTM targets
- Notebook: mini-batch training (batch 64) with `DataLoader`, a fixed seed, and automatic GPU use; a full headless run now takes about 3 minutes on a laptop CPU instead of about 70
- Add an offline pytest suite (leakage, fold boundaries, scaling round trip, metrics, seeded training, end-to-end benchmark) and run it in CI
- Add `chronos-forecasting` and `pytest` to `requirements.in` and regenerate the hash lock
- README: benchmark results and charts up front, walk-forward diagram, updated notebook numbers, design decisions and limitations

## [1.3.0] - 2026-10-18

- Fix train/test leakage: the last 30 test days of each stock were also in the training sequences; training windows now exclude them
- Fit each stock's MinMaxScaler on its training rows only, so the test window's min/max no longer leaks into training
- Build training windows per ticker instead of over the concatenated series, removing 15 windows that spanned two stocks
- Print a naive previous-close baseline MSE next to the model MSE for every stock
- Set `N` and `seq_length` once in the first cell; the evaluation cell no longer re-declares them, so changing `N` cannot bring the leak back
- Label the moving-average charts as min-max scaled and the feature chart as placeholder values, matching what the code plots
- Point the notebook's Colab badge at the lowercase repo name
- CI: add `permissions: contents: read`, install wheels only (`--only-binary :all:`) with hash checking (`--require-hashes`), and compile the notebook code on every push
- Split dependencies into `requirements.in` (direct pins) and a uv-generated hash-locked `requirements.txt`; Renovate updates them through its pip-compile manager
- Bump numpy to 2.5.4
- README: architecture diagram, measured results against the naive baseline, design decisions, limitations, uv setup; drop claims the code did not back (batch training, feature importance analysis, live-trading readiness)

## [1.2.1] - 2026-09-03

- Raise torch pin to 2.13.0 to resolve torch.jit.script memory corruption advisory (Dependabot alert #2)

## [1.2.0] - 2026-03-16

- Clear notebook outputs to reduce repo size
- Reclassify Jupyter notebooks as Python in language stats
- Add .gitignore

## [1.1.0] - 2025-08-21

- Add requirements.txt with project dependencies
- Update README with project overview, features, installation, and usage

## [1.0.0] - 2023-12-06

- LSTM stock price prediction in PyTorch
- Stocks: AAPL, MSFT, AMZN, TSLA (2020-2023)
- 6 features, 5-day sequences, 100 hidden units
- MinMaxScaler normalization, 150 epochs
- Moving average analysis (20-day, 50-day SMA)
- Jupyter notebook with Colab integration
