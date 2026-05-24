from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import json, os
from datetime import datetime

app = Flask(__name__)
CORS(app)

TICKERS = {'Netflix': 'NFLX', 'Walmart': 'WMT', 'Apple': 'AAPL'}
START_DATE = '2015-01-01'
FEATURES = [
    'Close_lag1', 'Close_lag2', 'Close_lag3',
    'MA_5', 'MA_10', 'MA_20', 'MA_50', 'Volatility',
    'RSI_14', 'MACD', 'MACD_signal', 'MACD_diff',
    'BB_upper', 'BB_lower', 'BB_width',
    'Volume_change', 'Price_momentum_5', 'Price_momentum_10',
    'Daily_return', 'Day_of_week', 'Month', 'Quarter',
    'Open', 'High', 'Low', 'Volume'
]

def download_data(ticker, start):
    """Download with multiple fallback methods"""
    # Method 1 — yf.download
    try:
        df = yf.download(
            ticker,
            start=start,
            auto_adjust=True,
            progress=False,
            timeout=30
        )
        if len(df) > 100:
            df = df.reset_index()
            # ✅ Fix: flatten multi-index columns properly
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    col[0] if col[1] == '' or col[1] == ticker
                    else f"{col[0]}_{col[1]}"
                    for col in df.columns
                ]
            else:
                df.columns = [
                    str(col).strip() for col in df.columns
                ]
            # ✅ Fix: rename 'index' or 'Datetime' to 'Date'
            if 'index' in df.columns:
                df.rename(columns={'index': 'Date'}, inplace=True)
            if 'Datetime' in df.columns:
                df.ren
