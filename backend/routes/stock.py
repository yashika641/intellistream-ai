from fastapi import APIRouter, Query
import yfinance as yf
from prophet import Prophet
import pandas as pd

router = APIRouter(tags=["Stock Dashboard"])

# Cache trained models (to avoid retraining every request)
prophet_models = {}


# ------------------------------------------
# Real-time stock data
# ------------------------------------------

def get_realtime_data(symbol: str):

    ticker = yf.Ticker(symbol)
    data = ticker.history(period="1d", interval="5m")

    if data.empty:
        return None

    latest = data.iloc[-1]
    previous = data.iloc[-2] if len(data) > 1 else latest

    current_price = float(latest["Close"])
    previous_price = float(previous["Close"])

    change = current_price - previous_price
    percent_change = (change / previous_price) * 100 if previous_price else 0

    return {
        "current_price": round(current_price, 2),
        "change": round(change, 2),
        "percent_change": round(percent_change, 2),
        "volume": int(latest["Volume"])
    }


# ------------------------------------------
# Train Prophet model
# ------------------------------------------

def train_prophet_model(symbol: str):

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1y")

    df = hist.reset_index()[["Date", "Close"]]
    df.columns = ["ds", "y"]
    df["ds"] = df["ds"].dt.tz_localize(None)

    model = Prophet(daily_seasonality="auto")
    model.fit(df)

    prophet_models[symbol] = model


# ------------------------------------------
# Generate forecast
# ------------------------------------------

def generate_forecast(symbol: str, days: int):

    if symbol not in prophet_models:
        train_prophet_model(symbol)

    model = prophet_models[symbol]

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)

    return forecast_df.to_dict(orient="records")


# ------------------------------------------
# Main API Endpoint
# ------------------------------------------

@router.get("/api/dashboard/{symbol}")
def get_stock(symbol: str, forecast_days: int = Query(30)):

    symbol = symbol.upper()

    realtime = get_realtime_data(symbol)

    if not realtime:
        return {"error": f"No data found for {symbol}"}

    forecast = generate_forecast(symbol, forecast_days)

    return {
        "symbol": symbol,
        "realtime_data": realtime,
        "forecast": forecast,
        "model": "Prophet"
    }