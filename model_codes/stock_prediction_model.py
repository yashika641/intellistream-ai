# =====================================================
# 📈 Prophet Stock Forecast Training Script
# =====================================================

import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =====================================================
# ⚙️ CONFIGURATION
# =====================================================

TICKER = "NFLX"        # Change to DIS, WBD, PARA etc.
TRAIN_MONTHS = 6
FORECAST_DAYS = 30
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# 📥 STEP 1: Fetch Historical Data
# =====================================================

print(f"📥 Fetching last {TRAIN_MONTHS} months of data for {TICKER}...")
data = yf.download(
    TICKER,
    period=f"{TRAIN_MONTHS}mo",
    interval="1d",
    auto_adjust=True,
    progress=False
)

if data.empty:
    raise ValueError("No data downloaded.")

data = data.reset_index()

# Force Close column to be 1D
close_series = data["Close"]

# If it's a DataFrame (multi-index), squeeze it
if isinstance(close_series, pd.DataFrame):
    close_series = close_series.squeeze()

df = pd.DataFrame({
    "ds": pd.to_datetime(data["Date"]),
    "y": pd.to_numeric(pd.Series(close_series))
})

print(f"✅ Downloaded {len(df)} rows")

# =====================================================
# 🔀 STEP 2: Train-Test Split (80/20)
# =====================================================

split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

print(f"Training size: {len(train_df)}")
print(f"Testing size : {len(test_df)}")

# =====================================================
# 🤖 STEP 3: Train Prophet Model
# =====================================================

model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False,
    changepoint_prior_scale=0.1
)

model.fit(train_df)

print("✅ Prophet model trained successfully")

# =====================================================
# 🔮 STEP 4: Forecast On Test Set
# =====================================================

future_test = model.make_future_dataframe(periods=len(test_df))
forecast_test = model.predict(future_test)

forecast_test = forecast_test[["ds", "yhat"]]

# Merge actual vs predicted
merged = pd.merge(test_df, forecast_test, on="ds", how="inner")

# =====================================================
# 📊 STEP 5: Evaluation Metrics
# =====================================================

mae = mean_absolute_error(merged["y"], merged["yhat"])
rmse = np.sqrt(mean_squared_error(merged["y"], merged["yhat"]))
mape = np.mean(np.abs((merged["y"] - merged["yhat"]) / merged["y"])) * 100

print("\n📊 MODEL PERFORMANCE")
print("=" * 40)
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")
print("=" * 40)

# =====================================================
# 🔮 STEP 6: Future Forecast (Next 30 Days)
# =====================================================

future = model.make_future_dataframe(periods=FORECAST_DAYS)
forecast = model.predict(future)

forecast_future = forecast.tail(FORECAST_DAYS)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

print("\n📅 Next 30 Day Forecast Sample:")
print(forecast_future.head())

# =====================================================
# 💾 STEP 7: Save Model
# =====================================================

model_path = os.path.join(MODEL_DIR, f"{TICKER}_prophet_model.pkl")

import pickle
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"\n💾 Model saved to {model_path}")

# =====================================================
# 📈 STEP 8: Plot Results
# =====================================================

plt.figure(figsize=(12, 6))
plt.plot(train_df["ds"], train_df["y"], label="Train")
plt.plot(test_df["ds"], test_df["y"], label="Test")
plt.plot(merged["ds"], merged["yhat"], label="Predicted", linestyle="--")
plt.legend()
plt.title(f"{TICKER} Prophet Forecast (Train/Test)")
plt.show()

# Prophet built-in plots
model.plot(forecast)
plt.title(f"{TICKER} - 30 Day Forecast")
plt.show()

print("\n🎉 Prophet Stock Forecast Training Complete!")