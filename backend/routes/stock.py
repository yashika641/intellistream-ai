"""
dashboard_routes.py

Provides:
- Real-time stock price
- Change (TC)
- Dashboard KPIs
- Prophet forecast
"""

from fastapi import APIRouter, Query
from prophet import Prophet
import pandas as pd
import yfinance as yf
from datetime import datetime

router = APIRouter(
    tags=["Stock Dashboard"]
)

# ==========================================
# Utility: Fetch Real-Time Data
# ==========================================

def get_realtime_data(symbol: str):
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
# Utility: Prophet Forecast
# ==========================================

def generate_prophet_forecast(symbol: str, days: int):
    """
    Generate Prophet forecast for next N days.
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1y")

    df = hist.reset_index()[["Date", "Close"]]
    df.columns = ["ds", "y"]
    df["ds"] = df["ds"].dt.tz_localize(None)
    
    model = Prophet(daily_seasonality='auto')
    model.fit(df)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days)

    return forecast_df.to_dict(orient="records")


# ==========================================
# Dashboard API Route
# ==========================================

@router.get("/api/dashboard/{symbol}")
def get_dashboard_data(
    symbol: str,
    forecast_days: int = Query(30, description="Number of forecast days")
):
    """
    Returns:
    - Real-time price
    - Change (TC)
    - Dashboard KPIs
    - Prophet forecast
    """

    realtime = get_realtime_data(symbol.upper())

    if realtime is None:
        return {"error": "No data available"}

    forecast = generate_prophet_forecast(symbol.upper(), forecast_days)

    response = {
        "symbol": symbol.upper(),
        "realtime_data": realtime,
        "kpis": {
            "current_price": realtime["current_price"],
            "change": realtime["change"],
            "percent_change": realtime["percent_change"],
            "volume": realtime["volume"]
        },
        "forecast": forecast,
        "model": "Prophet"
    }

    return response