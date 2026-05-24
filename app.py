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
                df.rename(columns={'Datetime': 'Date'}, inplace=True)
            if 'Date' not in df.columns and df.index.name == 'Date':
                df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'])
            print(f"Method 1 OK: {len(df)} rows, cols: {list(df.columns)}")
            return df
    except Exception as e:
        print(f"Method 1 failed: {e}")

    # Method 2 — Ticker.history
    try:
        t = yf.Ticker(ticker)
        df = t.history(
            start=start,
            auto_adjust=True
        )
        if len(df) > 100:
            df = df.reset_index()
            df.columns = [str(c).strip() for c in df.columns]
            if 'index' in df.columns:
                df.rename(columns={'index': 'Date'}, inplace=True)
            if 'Datetime' in df.columns:
                df.rename(columns={'Datetime': 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            print(f"Method 2 OK: {len(df)} rows")
            return df
    except Exception as e:
        print(f"Method 2 failed: {e}")

    return pd.DataFrame()

def add_features(df):
    df = df.copy().sort_values('Date').reset_index(drop=True)
    df['Close_lag1'] = df['Close'].shift(1)
    df['Close_lag2'] = df['Close'].shift(2)
    df['Close_lag3'] = df['Close'].shift(3)
    df['MA_5']       = df['Close'].rolling(5).mean()
    df['MA_10']      = df['Close'].rolling(10).mean()
    df['MA_20']      = df['Close'].rolling(20).mean()
    df['MA_50']      = df['Close'].rolling(50).mean()
    df['Volatility'] = df['Close'].rolling(5).std()
    df['RSI_14']     = ta.momentum.RSIIndicator(
                           df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD']        = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff']   = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df['Close'], window=20)
    df['BB_upper']  = bb.bollinger_hband()
    df['BB_lower']  = bb.bollinger_lband()
    df['BB_width']  = bb.bollinger_wband()
    df['Volume_change']     = df['Volume'].pct_change()
    df['Price_momentum_5']  = df['Close'] - df['Close'].shift(5)
    df['Price_momentum_10'] = df['Close'] - df['Close'].shift(10)
    df['Daily_return']      = df['Close'].pct_change()
    df['Day_of_week']       = df['Date'].dt.dayofweek
    df['Month']             = df['Date'].dt.month
    df['Quarter']           = df['Date'].dt.quarter
    df.dropna(inplace=True)
    return df

def forecast_sklearn(model, last_row, n_days, features):
    predictions = []
    current = last_row[features].copy()
    for _ in range(n_days):
        pred = model.predict(
            current.values.reshape(1, -1))[0]
        predictions.append(pred)
        current['Close_lag3'] = current['Close_lag2']
        current['Close_lag2'] = current['Close_lag1']
        current['Close_lag1'] = pred
    return predictions

def run_predictions():
    all_rows     = []
    metrics_rows = []

    for name, ticker in TICKERS.items():
        print(f"\n=== Processing {name} ({ticker}) ===")

        df = download_data(ticker, START_DATE)

        if df.empty or len(df) < 200:
            print(f"❌ Not enough data for {name}")
            continue

        if 'Date' not in df.columns:
            print(f"❌ Date column missing for {name}")
            print(f"Available columns: {list(df.columns)}")
            continue

        df['Date'] = pd.to_datetime(df['Date'])
        df = add_features(df)

        if len(df) < 100:
            print(f"❌ Not enough rows after features for {name}")
            continue

        split   = int(len(df) * 0.8)
        train   = df.iloc[:split]
        test    = df.iloc[split:]
        X_tr    = train[FEATURES]
        y_tr    = train['Close']
        X_te    = test[FEATURES]
        y_te    = test['Close']

        models = {
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'Linear Regression': LinearRegression(),
            'XGBoost': XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        }

        best_model = None
        best_r2    = -999
        best_name  = 'XGBoost'

        for model_name, model in models.items():
            try:
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)

                rmse = float(np.sqrt(
                    mean_squared_error(y_te, y_pred)))
                r2   = float(r2_score(y_te, y_pred))
                mape = float(np.mean(
                    np.abs((y_te.values - y_pred)
                           / y_te.values)) * 100)

                metrics_rows.append({
                    'Stock':    name,
                    'Model':    model_name,
                    'RMSE':     round(rmse, 4),
                    'R2_Score': round(r2, 4),
                    'MAPE_Pct': round(mape, 2)
                })

                for date, actual, pred in zip(
                    test['Date'].values,
                    y_te.values,
                    y_pred
                ):
                    all_rows.append({
                        'Date': pd.Timestamp(date).strftime(
                            '%Y-%m-%d'),
                        'Stock':           name,
                        'Type':            'Historical Prediction',
                        'Model':           model_name,
                        'Actual_Close':    round(float(actual), 2),
                        'Predicted_Close': round(float(pred), 2),
                        'RMSE':            round(rmse, 4),
                        'R2':              round(r2, 4),
                        'MAPE':            round(mape, 2),
                        'Last_Updated':    datetime.now().strftime(
                            '%Y-%m-%d %H:%M')
                    })

                if r2 > best_r2:
                    best_r2    = r2
                    best_model = model
                    best_name  = model_name

            except Exception as e:
                print(f"Model {model_name} failed: {e}")
                continue

        # 30-day forecast
        if best_model is not None:
            try:
                last_row  = df.iloc[-1]
                last_date = last_row['Date']
                future_preds = forecast_sklearn(
                    best_model, last_row, 30, FEATURES)
                future_dates = pd.bdate_range(
                    start=pd.Timestamp(last_date)
                          + pd.Timedelta(days=1),
                    periods=30
                )
                for date, pred in zip(future_dates, future_preds):
                    all_rows.append({
                        'Date':            str(date.date()),
                        'Stock':           name,
                        'Type':            'Future Forecast',
                        'Model':           best_name,
                        'Actual_Close':    0,
                        'Predicted_Close': round(float(pred), 2),
                        'RMSE':            round(best_r2, 4),
                        'R2':              round(best_r2, 4),
                        'MAPE':            0,
                        'Last_Updated':    datetime.now().strftime(
                            '%Y-%m-%d %H:%M')
                    })
            except Exception as e:
                print(f"Forecast failed for {name}: {e}")

        print(f"✅ {name} done")

    return all_rows, metrics_rows

@app.route('/')
def home():
    return jsonify({
        'status':  'running',
        'version': '2.0',
        'stocks':  list(TICKERS.keys()),
        'models':  ['KNN', 'Linear Regression', 'XGBoost']
    })

@app.route('/predictions')
def get_predictions():
    try:
        rows, _ = run_predictions()
        stock = request.args.get('stock')
        model = request.args.get('model')
        if stock:
            rows = [r for r in rows if r['Stock'] == stock]
        if model:
            rows = [r for r in rows if r['Model'] == model]
        resp = app.response_class(
            response=json.dumps(rows),
            status=200,
            mimetype='application/json'
        )
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def get_metrics():
    try:
        _, metrics = run_predictions()
        resp = app.response_class(
            response=json.dumps(metrics),
            status=200,
            mimetype='application/json'
        )
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/forecast')
def get_forecast():
    try:
        rows, _ = run_predictions()
        data = [r for r in rows
                if r['Type'] == 'Future Forecast']
        resp = app.response_class(
            response=json.dumps(data),
            status=200,
            mimetype='application/json'
        )
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/historical')
def get_historical():
    try:
        rows, _ = run_predictions()
        data = [r for r in rows
                if r['Type'] == 'Historical Prediction']
        resp = app.response_class(
            response=json.dumps(data),
            status=200,
            mimetype='application/json'
        )
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
