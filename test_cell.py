#!/usr/bin/env python
"""
Execute first cell of notebook 6
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_data
from strategy import EMAStrategy, TradeAnalyzer, apply_ml_filter
from ml_models import MLFeatureEngineer, XGBoostModel, LSTMModel, ModelEvaluator
from backtest import Backtester, BacktestVisualizer, generate_backtest_report
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")
