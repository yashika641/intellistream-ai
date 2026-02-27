# ==============================================
# 📈 FastAPI: Stock Prediction Routes
# ==============================================
import os

from fastapi import APIRouter,Request
from fastapi.middleware.cors import CORSMiddleware
from GoogleNews import GoogleNews
import pandas as pd
import plotly.graph_objects as go
import joblib
import datetime
import numpy as np
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---------- Load Models ----------
sentiment_model = joblib.load(os.path.join(BASE_DIR, "models", "sentiment_classifier.joblib"))
price_model = load_model(
    os.path.join(BASE_DIR, "models", "news_aware_netflix_model.keras")
)
volume_model = load_model(
    os.path.join(BASE_DIR, "models", "netflix_stock_volume_model.h5")
)
# ---------- FastAPI App ----------
router = APIRouter(prefix="/api", tags=["stock"])

# ---------- Utils ----------
def fetch_news_headlines(query="Netflix stock", num_headlines=10):
    gn = GoogleNews(lang='en', period='7d')
    gn.search(query)
    news = gn.result()[:num_headlines]
    return [n['title'] for n in news]

def analyze_sentiment(headlines):
    # Predict labels
    labels = [sentiment_model.predict([h])[0] for h in headlines]

    # Map labels to numeric values
    mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
    scores = [mapping.get(l, 0) for l in labels]  # default 0 if label not found

    avg_sentiment = float(np.mean(scores))
    return avg_sentiment


def predict_next_7_days(avg_sentiment):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    CSV_PATH = os.path.join(BASE_DIR, "docs", "aggregated_netflix_news_stock.csv")
    # Sequence lengths (as used in training)
    SEQ_LEN_PRICE = 7
    SEQ_LEN_VOLUME = 21
    HORIZON = 7

    # 1) Load and aggregate daily (same as training script)
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    # ensure sentiment_score exists
    if 'sentiment' in df.columns and 'sentiment_score' not in df.columns:
        mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
        df['sentiment_score'] = df['sentiment'].map(mapping).fillna(0.0)
    elif 'sentiment_score' not in df.columns:
        df['sentiment_score'] = 0.0

    daily = df.groupby('date').agg({
        'stock_open': 'first',
        'stock_close': 'last',
        'stock_high': 'max',
        'stock_low': 'min',
        'stock_volume': 'mean',
        'sentiment_score': 'mean'
    }).reset_index()

    # -----------------------------------
    # Price scalers (fit on train portion)
    # -----------------------------------
    features_price = ['stock_open', 'stock_high', 'stock_low', 'stock_volume', 'sentiment_score']
    targets_price = ['stock_close', 'stock_open', 'stock_high', 'stock_low']

    if len(daily) < SEQ_LEN_PRICE + 10:
        raise ValueError("Not enough history in CSV to form price sequences.")

    split_idx = int(0.8 * len(daily))
    scaler_X_price = MinMaxScaler().fit(daily[features_price].iloc[:split_idx].values)
    scaler_y_price = MinMaxScaler().fit(daily[targets_price].iloc[:split_idx].values)

    # prepare raw last-price-window and scale it
    last_price_raw = daily[features_price].values[-SEQ_LEN_PRICE:].astype(float)  # shape (SEQ_LEN_PRICE, n_feat)
    # replace last timestep sentiment with avg_sentiment (raw)
    sent_idx = features_price.index('sentiment_score')
    last_price_raw[-1, sent_idx] = avg_sentiment
    last_price_scaled = scaler_X_price.transform(last_price_raw)  # scaled window used as model input

    # -----------------------------------
    # Volume scalers & features (match your volume preprocessing)
    # -----------------------------------
    # Build daily volume features like in your volume script
    vol_daily = df.copy()
    vol_daily = vol_daily.groupby('date').agg({
        'stock_open': 'first',
        'stock_close': 'last',
        'stock_high': 'max',
        'stock_low': 'min',
        'stock_volume': 'sum',
        'headlines': (lambda x: ' '.join([str(v) for v in x]) ) if 'headlines' in df.columns else (lambda x: '')
    }).reset_index()

    vol_daily['volume_log'] = np.log1p(vol_daily['stock_volume'])
    lags = [1, 7, 14, 21]
    for lag in lags:
        vol_daily[f'lag_{lag}'] = vol_daily['volume_log'].shift(lag)
    for window in [7, 14]:
        vol_daily[f'roll_mean_{window}'] = vol_daily['volume_log'].rolling(window=window).mean()
        vol_daily[f'roll_std_{window}'] = vol_daily['volume_log'].rolling(window=window).std()
    vol_daily['day_of_week'] = vol_daily['date'].dt.dayofweek

    # expand day-of-week dummies (drop first)
    dow_dummies = pd.get_dummies(vol_daily['day_of_week'], prefix='dow', drop_first=True)
    vol_daily = pd.concat([vol_daily, dow_dummies], axis=1)

    # sentiment (VADER or existing)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    if 'headlines' in vol_daily.columns:
        vol_daily['sentiment_score'] = vol_daily['headlines'].fillna('').astype(str).apply(lambda x: sia.polarity_scores(x)['compound'])
    else:
        vol_daily['sentiment_score'] = 0.0
    vol_daily['sentiment_lag_1'] = vol_daily['sentiment_score'].shift(1)

    # Drop rows with NaNs created by rolling/shift
    vol_daily_clean = vol_daily.dropna().reset_index(drop=True)

    # Choose features used for volume model (match your training)
    features_volume = [c for c in vol_daily_clean.columns if c not in ['date', 'stock_volume', 'volume_log']]
    X_vol_all = vol_daily_clean[features_volume].astype(float).values
    y_vol_all = vol_daily_clean['volume_log'].astype(float).values.reshape(-1, 1)

    if len(X_vol_all) < SEQ_LEN_VOLUME + 10:
        raise ValueError("Not enough volume-history after rolling/shifts to form sequences.")

    split_idx_vol = int(0.8 * len(X_vol_all))
    scaler_X_vol = MinMaxScaler().fit(X_vol_all[:split_idx_vol])

    # Prepare last volume raw window and replace last sentiment with avg_sentiment
    last_vol_raw = X_vol_all[-SEQ_LEN_VOLUME:].copy()  # raw numerical features
    # If 'sentiment_score' in features_volume, replace last row value:
    if 'sentiment_score' in features_volume:
        sidx = features_volume.index('sentiment_score')
        # raw values -> replace
        last_vol_raw[-1, sidx] = avg_sentiment
    # scale last window
    last_vol_scaled = scaler_X_vol.transform(last_vol_raw)

    # -----------------------------------
    # Iterative forecasting loop
    # -----------------------------------
    preds = []
    dates = []
    base_date = daily['date'].iloc[-1].date()
    current_price_scaled = last_price_scaled.copy()  # scaled window
    current_price_raw = last_price_raw.copy()        # raw window (kept to update fields easily)

    current_vol_scaled = last_vol_scaled.copy()
    current_vol_raw = last_vol_raw.copy()

    for step in range(1, HORIZON + 1):
        forecast_date = base_date + np.timedelta64(step, 'D')
        # Price model input & predict
        price_input = current_price_scaled.reshape(1, SEQ_LEN_PRICE, len(features_price))
        pred_price_scaled = price_model.predict(price_input, verbose=0)
        # price_model outputs scaled targets (as in training), inverse-transform
        pred_price_raw = scaler_y_price.inverse_transform(pred_price_scaled.reshape(1, -1))[0]
        # map outputs
        pred_close = float(pred_price_raw[0])
        pred_open = float(pred_price_raw[1])
        pred_high = float(pred_price_raw[2])
        pred_low = float(pred_price_raw[3])

        # Volume model input & predict
        vol_input = current_vol_scaled.reshape(1, SEQ_LEN_VOLUME, current_vol_scaled.shape[1])
        pred_vol_log = volume_model.predict(vol_input, verbose=0).flatten()[0]  # assumed to be volume_log
        pred_volume = float(np.expm1(pred_vol_log))

        # Save prediction for this day
        preds.append({
            'date': pd.to_datetime(forecast_date).strftime('%Y-%m-%d'),
            'stock_open': pred_open,
            'stock_close': pred_close,
            'stock_high': pred_high,
            'stock_low': pred_low,
            'stock_volume': pred_volume
        })
        dates.append(forecast_date)

        # ---------------------------
        # Build next timestep raw rows (for price and volume windows)
        # ---------------------------
        # For price features: order ['stock_open','stock_high','stock_low','stock_volume','sentiment_score']
        next_price_raw = np.array([pred_open, pred_high, pred_low, pred_volume, avg_sentiment], dtype=float)
        # slide raw & scaled windows for price
        current_price_raw = np.vstack([current_price_raw[1:], next_price_raw])
        # scale the newly formed window using scaler_X_price
        current_price_scaled = scaler_X_price.transform(current_price_raw)

        # For volume features: create next raw row based on features_volume layout
        # We'll create next_vol_raw by copying last raw row and replacing numeric fields we can infer:
        next_vol_raw = current_vol_raw[-1].copy()  # base from last raw
        # Try to update meaningful columns if they exist in features_volume
        fv = features_volume
        if 'stock_open' in fv:
            next_vol_raw[fv.index('stock_open')] = pred_open
        if 'stock_high' in fv:
            next_vol_raw[fv.index('stock_high')] = pred_high
        if 'stock_low' in fv:
            next_vol_raw[fv.index('stock_low')] = pred_low
        if 'stock_volume' in fv:
            # Many volume feature sets used 'stock_volume' raw; if present, set to predicted_volume
            next_vol_raw[fv.index('stock_volume')] = pred_volume
        if 'sentiment_score' in fv:
            next_vol_raw[fv.index('sentiment_score')] = avg_sentiment
        # For lags/rolling features which depend on previous values, the model was trained with them populated.
        # Here we do a simple approximation: keep existing lag/rolling numeric values from last row (they will be slightly stale).
        # Slide raw & scaled windows for volume
        current_vol_raw = np.vstack([current_vol_raw[1:], next_vol_raw])
        current_vol_scaled = scaler_X_vol.transform(current_vol_raw)

    # build result DataFrame
    df_pred = pd.DataFrame(preds)
    return df_pred


# ---------- Routes ----------
@router.get("/predict-stock")
def predict_stock(request: Request):
    headlines = fetch_news_headlines()
    avg_sentiment = analyze_sentiment(headlines)
    predictions = predict_next_7_days(avg_sentiment)

    # store in app.state
    request.app.state.latest_sentiment = avg_sentiment
    request.app.state.latest_predictions = predictions

    return {
        "headlines": headlines,
        "avg_sentiment": avg_sentiment,
        "predictions": predictions.to_dict(orient="records")
    }

# -------------------------
# Plot-stock-metrics route
# -------------------------
@router.get("/plot-stock-metrics")
def plot_stock_metrics(request: Request):
    if not hasattr(request.app.state, "latest_predictions") or request.app.state.latest_predictions is None:
        return JSONResponse({"error": "Please call /predict-stock first"}, status_code=400)

    df_pred = request.app.state.latest_predictions
    avg_sentiment = request.app.state.latest_sentiment

    # Use 'stock_close' as predicted price, 'stock_volume' as volume
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_pred["date"], y=df_pred["stock_close"],
        mode='lines+markers', name='Predicted Price', line=dict(color='cyan', width=3)
    ))
    fig.add_trace(go.Bar(
        x=df_pred["date"], y=df_pred["stock_volume"],
        name='Predicted Volume', opacity=0.5, yaxis='y2'
    ))

    fig.update_layout(
        title=f"📈 Netflix 7-Day Stock Prediction | Avg Sentiment: {avg_sentiment:.2f}",
        xaxis_title="Date",
        yaxis=dict(title="Price (USD)"),
        yaxis2=dict(title="Volume", overlaying='y', side='right'),
        template="plotly_dark",
        legend=dict(x=0, y=1.1, orientation="h")
    )

    return JSONResponse(fig.to_json())