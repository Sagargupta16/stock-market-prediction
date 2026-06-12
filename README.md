# 📈 Stock Market Prediction with LSTM

A comprehensive machine learning project that uses Long Short-Term Memory (LSTM) neural networks to predict stock prices for major technology companies. This project demonstrates the application of deep learning techniques in financial forecasting.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sagargupta16/Stock-market-prediction/blob/main/SMP.ipynb)

## 🎯 Project Overview

This project implements an LSTM-based neural network to predict stock prices using historical market data. The model is trained on data from four major tech stocks: Apple (AAPL), Microsoft (MSFT), Amazon (AMZN), and Tesla (TSLA).

### Key Features

- **Multi-stock prediction**: Supports prediction for multiple stocks simultaneously
- **LSTM neural network**: Utilizes deep learning for time series forecasting
- **Real-time data fetching**: Uses Yahoo Finance API for up-to-date stock data
- **Comprehensive visualization**: Includes actual vs predicted price comparisons
- **Technical analysis**: Moving averages and feature importance analysis
- **Performance metrics**: Mean Squared Error evaluation for model accuracy

## 🚀 Getting Started

### Prerequisites

Before running this project, make sure you have the following installed:

```bash
pip install yfinance pandas torch scikit-learn matplotlib numpy
```

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Sagargupta16/Stock-market-prediction.git
cd Stock-market-prediction
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Open the Jupyter notebook:

```bash
jupyter notebook SMP.ipynb
```

## 📊 Model Architecture

### LSTM Network Structure

- **Input Layer**: 6 features (Open, High, Low, Close, Volume, Average)
- **LSTM Layer**: 100 hidden units
- **Output Layer**: 1 unit (predicted closing price)
- **Sequence Length**: 5 days
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Mean Squared Error (MSE)

### Data Processing Pipeline

1. **Data Fetching**: Historical stock data from Yahoo Finance (2020-2023)
2. **Feature Engineering**: Calculation of average price from high and low
3. **Normalization**: MinMaxScaler for feature scaling
4. **Sequence Creation**: Time series sequences for LSTM input
5. **Train/Test Split**: Last 30 days reserved for testing

## 📈 Results and Analysis

### Stock Performance Metrics

The model evaluates performance using Mean Squared Error (MSE) for each stock:

- **AAPL**: Apple Inc.
- **MSFT**: Microsoft Corporation
- **AMZN**: Amazon.com Inc.
- **TSLA**: Tesla Inc.

### Visualization Features

- **Price Prediction Charts**: Actual vs predicted prices comparison
- **Moving Averages**: 20-day and 50-day simple moving averages
- **Feature Importance**: Analysis of input feature contributions

## 🛠️ Usage

### Running the Prediction Model

1. **Data Preparation**:

```python
# Fetch and preprocess stock data
data_aapl, scaler_aapl = fetch_preprocess('AAPL', '2020-01-01', '2023-01-01')
```

2. **Model Training**:

```python
# Train the LSTM model
model = LSTM()
# Training loop with 150 epochs
```

3. **Prediction and Evaluation**:

```python
# Evaluate model performance
actual_prices, predicted_prices = evaluate_model(test_sequences, scaler, model, 'AAPL')
```

### Customization Options

- **Stock Symbols**: Modify the stock symbols to analyze different companies
- **Date Range**: Adjust start and end dates for different time periods
- **Model Parameters**: Tune LSTM hidden layer size, sequence length, and learning rate
- **Training Epochs**: Modify the number of training iterations

## 📁 Project Structure

```text
Stock-market-prediction/
├── SMP.ipynb           # Main Jupyter notebook with complete implementation
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
├── CHANGELOG.md        # Release history
├── SECURITY.md         # Security policy
└── LICENSE             # MIT license
```

## 🔧 Technical Details

### Dependencies

- **PyTorch**: Deep learning framework for LSTM implementation
- **yfinance**: Yahoo Finance API for stock data retrieval
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning utilities and preprocessing
- **matplotlib**: Data visualization and plotting
- **numpy**: Numerical computing support

### Model Training Process

1. **Data Normalization**: Scales all features to [0,1] range
2. **Sequence Generation**: Creates overlapping sequences for time series learning
3. **Batch Training**: Processes sequences in batches for efficient learning
4. **Hidden State Reset**: Resets LSTM hidden states for each sequence
5. **Gradient Optimization**: Uses Adam optimizer for parameter updates

## 📊 Performance Considerations

- **Training Time**: Approximately 150 epochs for convergence
- **Memory Usage**: Depends on sequence length and batch size
- **Prediction Accuracy**: Varies by stock volatility and market conditions
- **Real-time Capability**: Can be adapted for live trading scenarios

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Areas for Improvement

- Integration of additional technical indicators
- Implementation of other neural network architectures (GRU, Transformer)
- Portfolio optimization features
- Real-time prediction API
- Enhanced visualization dashboard

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact

**Sagar Gupta** - [GitHub Profile](https://github.com/Sagargupta16)

Project Link: [https://github.com/Sagargupta16/Stock-market-prediction](https://github.com/Sagargupta16/Stock-market-prediction)

## ⚠️ Disclaimer

This project is for educational and research purposes only. The predictions made by this model should not be considered as financial advice. Always consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.

## 🙏 Acknowledgments

- Yahoo Finance for providing free access to historical stock data
- PyTorch community for excellent deep learning framework
- Financial data science community for inspiration and best practices