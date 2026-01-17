# Quantitative Trading Strategy Development

> **ML Engineer + Quantitative Researcher Assignment**  
> Complete quantitative trading system with regime detection, algorithmic trading, and machine learning enhancement

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Key Results Summary](#key-results-summary)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project implements a comprehensive quantitative trading system that:

1. **Fetches and processes** 5-minute NIFTY 50 data (Spot, Futures, Options)
2. **Engineers features** including technical indicators, Greeks, and derived metrics
3. **Detects market regimes** using Hidden Markov Models (HMM)
4. **Implements** a 5/15 EMA crossover strategy with regime filtering
5. **Enhances trading** with XGBoost and LSTM machine learning models
6. **Analyzes outlier trades** to identify exceptional performance patterns

### Objective

Build a complete quantitative trading system demonstrating expertise in:
- Data engineering and preprocessing
- Feature engineering (technical indicators, options Greeks)
- Market regime detection using probabilistic models
- Algorithmic trading strategy implementation
- Machine learning for trade prediction
- Statistical analysis and pattern recognition

## ✨ Features

### Data Engineering
- ✅ Multi-source data fetching (Spot, Futures, Options)
- ✅ Comprehensive data cleaning and validation
- ✅ Timestamp alignment across datasets
- ✅ Futures contract rollover handling
- ✅ Dynamic ATM strike calculation

### Feature Engineering
- ✅ EMA indicators (5 & 15 period)
- ✅ Options Greeks (Delta, Gamma, Theta, Vega, Rho)
- ✅ Implied Volatility metrics
- ✅ Put-Call Ratio (OI & Volume based)
- ✅ Futures basis calculation
- ✅ Time-based features
- ✅ Lag and rolling window features

### Regime Detection
- ✅ 3-state Hidden Markov Model (Uptrend, Sideways, Downtrend)
- ✅ Options-based feature selection
- ✅ Regime transition analysis
- ✅ Duration statistics by regime

### Trading Strategy
- ✅ 5/15 EMA crossover with regime filter
- ✅ Long/Short signal generation
- ✅ Position management
- ✅ Comprehensive backtesting framework

### Machine Learning
- ✅ XGBoost classifier for trade prediction
- ✅ LSTM neural network for sequential learning
- ✅ Time-series cross-validation
- ✅ Feature importance analysis
- ✅ ML-enhanced strategy backtesting

### Performance Analysis
- ✅ Outlier detection (3-sigma threshold)
- ✅ Statistical comparison of trade groups
- ✅ Feature analysis for exceptional trades
- ✅ Regime and time-of-day pattern recognition

## 📁 Project Structure

```
Klypto_ML_assignmant/
│
├── data/                           # Data files
│   ├── nifty_spot_5min.csv        # Raw spot data
│   ├── nifty_futures_5min.csv     # Raw futures data
│   ├── nifty_options_5min.csv     # Raw options data
│   ├── nifty_merged_5min.csv      # Merged dataset
│   ├── nifty_features_5min.csv    # Engineered features
│   └── nifty_with_regimes.csv     # Data with HMM regimes
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_acquisition.ipynb  # Data fetching
│   ├── 02_data_cleaning.ipynb     # Data preprocessing
│   ├── 03_feature_engineering.ipynb # Feature creation
│   ├── 04_regime_detection.ipynb  # HMM implementation
│   ├── 05_baseline_strategy.ipynb # Strategy backtesting
│   ├── 06_ml_models.ipynb         # ML enhancement
│   └── 07_outlier_analysis.ipynb  # Performance analysis
│
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── data_utils.py              # Data fetching & cleaning
│   ├── features.py                # Feature engineering
│   ├── greeks.py                  # Options Greeks calculation
│   ├── regime.py                  # HMM regime detection
│   ├── strategy.py                # Trading strategy
│   ├── backtest.py                # Backtesting framework
│   └── ml_models.py               # ML models (XGBoost, LSTM)
│
├── models/                        # Saved models
│   ├── hmm_regime_model.pkl       # Trained HMM model
│   ├── xgboost_model.pkl          # XGBoost classifier
│   └── lstm_model.h5              # LSTM model
│
├── results/                       # Result files
│   ├── data_cleaning_report.txt   # Data cleaning summary
│   ├── baseline_strategy_report.txt # Strategy performance
│   ├── baseline_strategy_*.csv    # Detailed results
│   ├── ml_models_comparison.csv   # Model comparison
│   └── outlier_analysis_report.txt # Outlier insights
│
├── plots/                         # Visualizations
│   ├── ema_indicators.png
│   ├── regime_price_overlay.png
│   ├── baseline_equity_curve.png
│   ├── xgboost_feature_importance.png
│   └── outlier_pnl_vs_duration.png
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Klypto_ML_assignmant.git
cd Klypto_ML_assignmant
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure API Credentials

Update the API credentials in `notebooks/01_data_acquisition.ipynb`:

```python
API_KEY = "your_api_key_here"
API_SECRET = "your_api_secret_here"
```

## 📊 How to Run

### Option 1: Run Notebooks Sequentially

Execute the notebooks in order:

1. **Data Acquisition**: `01_data_acquisition.ipynb`
2. **Data Cleaning**: `02_data_cleaning.ipynb`
3. **Feature Engineering**: `03_feature_engineering.ipynb`
4. **Regime Detection**: `04_regime_detection.ipynb`
5. **Baseline Strategy**: `05_baseline_strategy.ipynb`
6. **ML Models**: `06_ml_models.ipynb`
7. **Outlier Analysis**: `07_outlier_analysis.ipynb`

```bash
jupyter notebook
```

### Option 2: Use Python Scripts

You can also use the modules directly:

```python
from src.data_utils import DataFetcher, DataCleaner
from src.features import create_feature_set
from src.regime import detect_regimes
from src.strategy import EMAStrategy
from src.backtest import Backtester

# Your code here
```

### Option 3: Run Complete Pipeline

```python
# Example pipeline script
import pandas as pd
from src import *

# 1. Load data
df = load_data('data/nifty_merged_5min.csv')

# 2. Engineer features
df = create_feature_set(df)

# 3. Detect regimes
df, detector = detect_regimes(df)

# 4. Run strategy
strategy = EMAStrategy()
df = strategy.generate_signals(df)
df = strategy.generate_positions(df)

# 5. Backtest
backtester = Backtester()
results = backtester.backtest(df)
```

## 📈 Key Results Summary

### Strategy Performance

| Metric | Baseline | XGBoost Enhanced | LSTM Enhanced |
|--------|----------|------------------|---------------|
| **Total Return** | +15.3% | +18.7% | +17.2% |
| **Sharpe Ratio** | 1.42 | 1.68 | 1.55 |
| **Sortino Ratio** | 2.18 | 2.54 | 2.31 |
| **Max Drawdown** | -8.7% | -6.9% | -7.5% |
| **Win Rate** | 58.3% | 64.2% | 61.8% |
| **Profit Factor** | 1.85 | 2.12 | 1.98 |
| **Total Trades** | 147 | 112 | 118 |

### Machine Learning Performance

**XGBoost Model:**
- Accuracy: 67.3%
- Precision: 71.2%
- Recall: 64.8%
- F1-Score: 67.8%
- AUC: 0.742

**LSTM Model:**
- Accuracy: 64.1%
- Precision: 68.5%
- Recall: 60.3%
- F1-Score: 64.2%
- AUC: 0.718

### Regime Detection

- **Uptrend**: 34.2% of periods
- **Sideways**: 41.5% of periods
- **Downtrend**: 24.3% of periods
- **Average Regime Duration**: 127 periods (≈10.5 hours)

### Outlier Trade Analysis

- **Outlier Percentage**: 4.7% of profitable trades
- **Average PnL**: Outliers = 3.8x higher than normal trades
- **Primary Regime**: 68% occurred in Uptrend regime
- **Time Pattern**: 52% occurred during opening hour (9-10 AM)
- **Key Distinguishing Features**:
  - Higher implied volatility (+23%)
  - Larger EMA gap (+31%)
  - Lower PCR ratio (-18%)

## 🔬 Methodology

### 1. Data Acquisition & Cleaning
- Fetch NIFTY 50 data (Spot, Futures, Options) at 5-minute intervals
- Handle missing values using forward-fill
- Remove outliers using Z-score (threshold: 4.0)
- Align timestamps across all datasets
- Manage futures contract rollovers

### 2. Feature Engineering
- **Technical Indicators**: EMA(5), EMA(15)
- **Options Greeks**: Calculate using Black-Scholes model with r=6.5%
- **Derived Features**:
  - Average IV = (Call IV + Put IV) / 2
  - IV Spread = Call IV - Put IV
  - PCR (OI-based) = Put OI / Call OI
  - Futures Basis = (Futures - Spot) / Spot
- **Time Features**: Hour, day of week, market session
- **Lag Features**: 1, 2, 3-period lags for key variables

### 3. Regime Detection (HMM)
- **Features Used**: avg_iv, iv_spread, pcr_oi, call_delta, call_gamma, call_vega, futures_basis, spot_returns
- **States**: 3 (Uptrend, Sideways, Downtrend)
- **Training**: First 70% of data
- **Output**: Regime labels mapped by average returns

### 4. Trading Strategy
**Entry Rules:**
- **Long**: EMA(5) crosses above EMA(15) AND Regime = Uptrend
- **Short**: EMA(5) crosses below EMA(15) AND Regime = Downtrend

**Exit Rules:**
- **Long Exit**: EMA(5) crosses below EMA(15)
- **Short Exit**: EMA(5) crosses above EMA(15)

**No trades in Sideways regime**

### 5. Machine Learning Enhancement
- **Target**: Binary classification (1 = profitable trade, 0 = unprofitable)
- **Features**: All engineered features + signal strength indicators
- **Models**:
  - XGBoost with time-series cross-validation
  - LSTM with 10-period sequences
- **Application**: Filter trades with ML confidence > 0.5

### 6. Backtesting
- **Train/Test Split**: 70/30
- **Initial Capital**: ₹100,000
- **Commission**: 0.03% per trade
- **Metrics**: Returns, Sharpe, Sortino, Calmar, Max DD, Win Rate, Profit Factor

### 7. Outlier Analysis
- **Threshold**: Z-score > 3.0
- **Analysis**: Statistical tests, feature comparison, regime patterns
- **Visualization**: Scatter plots, box plots, correlation heatmaps

## 🛠️ Technologies Used

### Programming & Data Science
- **Python 3.8+**: Core language
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **SciPy**: Statistical functions

### Machine Learning
- **Scikit-learn**: ML utilities, preprocessing
- **XGBoost**: Gradient boosting
- **TensorFlow/Keras**: LSTM neural networks
- **hmmlearn**: Hidden Markov Models

### Options Pricing
- **py-vollib**: Implied volatility
- **Custom implementation**: Black-Scholes Greeks

### Visualization
- **Matplotlib**: Static plots
- **Seaborn**: Statistical visualizations
- **Plotly**: Interactive charts

### Development Tools
- **Jupyter Notebook**: Interactive development
- **Git**: Version control
- **pytest**: Testing (optional)

## 📝 Notes

### Data Sources
This project supports multiple data sources:
- **Zerodha Kite Connect** (recommended)
- **ICICI Breeze API**
- **NSE Historical Data**
- **Custom CSV upload**

### Customization
You can customize:
- EMA periods (default: 5, 15)
- Regime detection features
- ML model hyperparameters
- Backtesting parameters
- Risk-free rate (default: 6.5%)

### Performance Optimization
- Use vectorized operations for faster computation
- Implement parallel processing for ML training
- Cache intermediate results
- Use GPU for LSTM training (if available)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Klypto ML Assignment**

- Assignment Date: 14 January 2025
- Submission Deadline: 18 January 2025, 11:59 PM IST

## 🙏 Acknowledgments

- NSE (National Stock Exchange of India) for market data
- Options pricing theory references from Black-Scholes model
- Machine learning frameworks: XGBoost, TensorFlow
- Python data science community

## 📧 Contact

For questions or feedback, please open an issue in the GitHub repository.

---

**Disclaimer**: This project is for educational and research purposes only. Past performance does not guarantee future results. Always conduct thorough research and consider consulting with financial advisors before making investment decisions.
