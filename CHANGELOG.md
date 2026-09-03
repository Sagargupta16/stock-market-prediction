# Changelog

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
