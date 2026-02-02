import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Dummy model ve scaler (hiç kurulum yapmadan çalışacak)
try:
    model = joblib.load("ml_model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    model = LinearRegression()
    model.coef_ = np.array([0,0,0,0,0])
    model.intercept_ = 0
    joblib.dump(model,"ml_model.pkl")
    
    scaler = StandardScaler()
    scaler.mean_ = np.array([0,0,0,0,0])
    scaler.scale_ = np.array([1,1,1,1,1])
    scaler.var_ = np.array([1,1,1,1,1])
    joblib.dump(scaler,"scaler.pkl")

def compute_rsi(prices, period=14):
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rsi = 100 - (100 / (1 + ma_up/ma_down))
    return rsi

def predict_price_ml(symbol, market):
    ticker = symbol + ".IS" if market == "BIST" else symbol
    df = yf.download(ticker, period="1y", interval="1d")
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["RSI14"] = compute_rsi(df["Close"], 14)
    df = df.dropna()
    last_row = df.iloc[-1][["Close","MA5","MA20","RSI14","Volume"]].values.reshape(1,-1)
    last_scaled = scaler.transform(last_row)
    predicted = model.predict(last_scaled)[0]
    current_price = df["Close"].iloc[-1]
    return round(float(predicted),2), round(float(current_price),2)
