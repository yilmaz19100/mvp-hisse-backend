from fastapi import FastAPI
from predictor_ml import predict_price_ml
import datetime
import pandas as pd
import yfinance as yf

app = FastAPI(title="5 Günlük Hisse Tahmin API - ML V3")

@app.get("/predict_ml")
def predict_ml(symbol: str, market: str):
    pred, current = predict_price_ml(symbol, market)
    change_percent = round(((pred - current)/current)*100,2)
    return {
        "symbol": symbol,
        "market": market,
        "current_price": current,
        "predicted_price": pred,
        "change_percent": change_percent,
        "date": datetime.datetime.now().isoformat()
    }

@app.get("/real_price")
def real_price(symbol: str, market: str, date: str):
    ticker = symbol + ".IS" if market == "BIST" else symbol
    start = pd.to_datetime(date)
    df = yf.download(ticker, start=start, period="10d")
    if len(df) < 5:
        return {"error": "Yeterli veri yok"}
    real = round(float(df["Close"].iloc[4]),2)
    return {"real_price": real}
