"""
dashboard_routes.py

Provides:
- Real-time stock price
- Change (TC)
- Dashboard KPIs
- Prophet forecast
"""

from fastapi import APIRouter
import pandas as pd
from datetime import datetime

router = APIRouter(tags=["Stock Dashboard"])

# ==========================================
# Caches
# ==========================================

prophet_models = {}
dashboard_cache = {}

# ==========================================
# Utility: Fetch Real-Time Data
# ==========================================

def get_realtime_data(symbol: str):
    import yfinance as yf
    
    try:
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
        volume = int(latest["Volume"])

        return {
            "current_price": round(current_price, 2),
            "change": round(change, 2),
            "percent_change": round(percent_change, 2),
            "volume": volume
        }

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


# ==========================================
# Prophet Training
# ==========================================

def train_prophet_model(symbol: str):
    from prophet import Prophet
    import yfinance as yf
    

    global prophet_models

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1y")

    df = hist.reset_index()[["Date", "Close"]]
    df.columns = ["ds", "y"]
    df["ds"] = df["ds"].dt.tz_localize(None)

    model = Prophet(daily_seasonality="auto")
    model.fit(df)

    prophet_models[symbol] = model


def generate_prophet_forecast(symbol: str, days: int):

    if symbol not in prophet_models:
        train_prophet_model(symbol)

    model = prophet_models[symbol]

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)

    return forecast_df.to_dict(orient="records")


# ==========================================
# Batch Job (Scheduler)
# ==========================================

def run_stock_dashboard_batch():

    global dashboard_cache

    print("📊 Updating stock dashboard...")

    symbols = ["NFLX", "AMZN", "TSLA", "AAPL", "DIS"]

    results = {}

    for symbol in symbols:

        realtime = get_realtime_data(symbol)

        if realtime is None:
            continue

        forecast = generate_prophet_forecast(symbol, 30)

        results[symbol] = {
            "symbol": symbol,
            "realtime_data": realtime,
            "kpis": realtime,
            "forecast": forecast,
            "model": "Prophet"
        }

    dashboard_cache = {
        "updated_at": datetime.utcnow(),
        "stocks": results
    }

    print("✅ Stock dashboard updated")


# ==========================================
# Dashboard API Route
# ==========================================

@router.get("/api/dashboard")
def get_dashboard():

    if not dashboard_cache:
        return {"message": "Dashboard initializing..."}

    return dashboard_cache