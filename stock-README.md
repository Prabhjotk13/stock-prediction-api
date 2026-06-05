# 📈 Stock Market Prediction — ML Model + REST API + Tableau Dashboard

An end-to-end data science project that predicts stock prices for **Apple, Netflix, and Walmart** using machine learning, serves predictions via a **Flask REST API**, and visualizes them in an **interactive Tableau dashboard**.

---

## 🖥️ Live Dashboard

🔗 [View on Tableau Public](https://public.tableau.com/app/profile/prabhjot.kaur4890/viz/StockMarketPredictionDashboard/Dashboard2)

![Dashboard Preview](tableau-dashboard-preview.png)

---

## 🏗️ Project Architecture

```
Stock Data (yfinance)
        ↓
Feature Engineering (RSI, MACD, Bollinger Bands, MA, Momentum)
        ↓
ML Models (KNN → Linear Regression → XGBoost → LSTM)
        ↓
Flask REST API  ──→  Tableau Dashboard
```

---

## 📁 Repository Structure

```
stock-prediction-api/
│
├── notebook/
│   └── Advanced_Stock_Market_Prediction.ipynb   # Full ML pipeline
│
├── app.py                                        # Flask REST API
├── requirements.txt                              # Python dependencies
├── stock_wdc.html                                # Web client
├── tableau-dashboard-preview.png                 # Dashboard screenshot
└── README.md
```

---

## 🤖 Machine Learning Pipeline

### Data
- **Source:** Yahoo Finance via `yfinance` (real-time, no API key needed)
- **Stocks:** Apple (AAPL), Netflix (NFLX), Walmart (WMT)
- **Period:** January 2015 → Present
- **Split:** 80% training / 20% testing (chronological)

### Feature Engineering (26 features)
| Category | Features |
|---|---|
| Lag Features | Close_lag1, Close_lag2, Close_lag3 |
| Moving Averages | MA_5, MA_10, MA_20, MA_50 |
| Momentum | RSI_14, MACD, MACD_signal, MACD_diff, Price_momentum_5/10 |
| Volatility | Bollinger Bands (upper, lower, width), Rolling Std |
| Volume | Volume, Volume_change |
| Calendar | Day of week, Month, Quarter |
| OHLCV | Open, High, Low |

### Models Trained
| Model | Details |
|---|---|
| **KNN Regressor** | k=5 neighbors |
| **Linear Regression** | Baseline model |
| **XGBoost** | Hyperparameter tuned via GridSearchCV (best performer) |
| **LSTM** | Deep learning model for sequence prediction |

### Evaluation Metrics
- RMSE (Root Mean Squared Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

---

## 🔌 Flask REST API

The trained XGBoost model is served as a REST API using Flask, enabling real-time predictions and integration with Tableau.

### Run Locally
```bash
pip install -r requirements.txt
python app.py
```

---

## 📊 Tableau Dashboard Features

- **Highest close price** for Apple, Netflix, and Walmart
- **Predicted close price for the next 30 days** (line chart)
- **Top 10 days with highest stock price** per company (bar charts)
- **Interactive filters** by stock and date range

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Collection | `yfinance` |
| Data Processing | `pandas`, `numpy` |
| Feature Engineering | `ta` (Technical Analysis library) |
| Machine Learning | `scikit-learn`, `XGBoost` |
| Deep Learning | `TensorFlow / Keras (LSTM)` |
| API | `Flask` |
| Visualization | `Tableau Public`, `matplotlib` |

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/Prabhjotk13/stock-prediction-api.git
cd stock-prediction-api

# Install dependencies
pip install -r requirements.txt

# Run the API
python app.py
```

---

## 👤 About Me

Fresher Data Analyst passionate about building end-to-end data projects.
Connect with me on [LinkedIn](https://www.linkedin.com/in/prabhjot-kaur) *(update with your LinkedIn URL)*
