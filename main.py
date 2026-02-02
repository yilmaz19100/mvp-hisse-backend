from fastapi import FastAPI
from predictor_ml import predict_price_ml
import datetime

app = FastAPI()

@app.get("/predict_ml")
def predict_ml(symbol: str, market: str):
    pred, current = predict_price_ml(symbol, market)
    
    # ⚡ Sıfıra bölme kontrolü
    if current == 0:
        change_percent = 0.0
    else:
        change_percent = round(((pred - current)/current)*100,2)
    
    return {
        "symbol": symbol,
        "market": market,
        "current_price": current,
        "predicted_price": pred,
        "change_percent": change_percent,
        "date": str(datetime.datetime.now())
    }
