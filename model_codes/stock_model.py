# ===========================================
# 📈 LSTM Stock Forecast with Headlines + Sentiment
# ===========================================

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout #type: ignore
from tensorflow.keras.callbacks import EarlyStopping #type: ignore
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Load data
# -----------------------------
df = pd.read_csv(r"C:\Users\palya\Desktop\intellistream\intellistream-ai\docs\aggregated_netflix_news_stock.csv")
df.columns = [c.strip().lower() for c in df.columns]
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)

# -----------------------------
# 2️⃣ Sentiment analysis + BERT embeddings
# -----------------------------
analyzer = SentimentIntensityAnalyzer()
model_bert = SentenceTransformer('all-MiniLM-L6-v2')

tqdm.pandas(desc="Processing headlines")

# Compute sentiment for each headline
df['sentiment_score'] = df['headlines'].progress_apply(
    lambda x: analyzer.polarity_scores(str(x))['compound']
)

# Aggregate per day
daily = df.groupby('date').agg({
    'stock_open': 'first',
    'stock_close': 'last',
    'stock_high': 'max',
    'stock_low': 'min',
    'stock_volume': 'mean',
    'headlines': list,
    'sentiment_score': 'mean'
}).reset_index()

# Compute average BERT embedding per day
def get_avg_bert(headlines):
    embeddings = model_bert.encode(headlines)
    return embeddings.mean(axis=0)

daily['bert_embedding'] = daily['headlines'].progress_apply(get_avg_bert)

# -----------------------------
# 3️⃣ Combine features
# -----------------------------
# Convert bert_embedding to separate columns
bert_dim = daily['bert_embedding'][0].shape[0]
bert_df = pd.DataFrame(daily['bert_embedding'].to_list(), columns=[f'bert_{i}' for i in range(bert_dim)])
daily = pd.concat([daily, bert_df], axis=1)
daily.drop(['headlines', 'bert_embedding'], axis=1, inplace=True)

# Lagged features + rolling averages
SEQ_LEN = 7
feature_cols = ['stock_open', 'stock_high', 'stock_low', 'stock_volume', 'sentiment_score'] + [f'bert_{i}' for i in range(bert_dim)]
target_cols_price = ['stock_close', 'stock_open', 'stock_high', 'stock_low']
target_cols_volume = ['stock_volume']

# -----------------------------
# 4️⃣ Scaling
# -----------------------------
scaler_X = MinMaxScaler()
scaler_y_price = MinMaxScaler()
scaler_y_vol = MinMaxScaler()

scaled_features = scaler_X.fit_transform(daily[feature_cols])
scaled_targets_price = scaler_y_price.fit_transform(daily[target_cols_price])
scaled_targets_vol = scaler_y_vol.fit_transform(daily[target_cols_volume])

# -----------------------------
# 5️⃣ Sequence creation
# -----------------------------
def create_sequences(X, y, seq_len=SEQ_LEN):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

X_price, y_price = create_sequences(scaled_features, scaled_targets_price)
X_vol, y_vol = create_sequences(scaled_features, scaled_targets_vol)

# -----------------------------
# 6️⃣ Train-test split
# -----------------------------
split = int(0.8 * len(X_price))
X_train_price, X_test_price = X_price[:split], X_price[split:]
y_train_price, y_test_price = y_price[:split], y_price[split:]

X_train_vol, X_test_vol = X_vol[:split], X_vol[split:]
y_train_vol, y_test_vol = y_vol[:split], y_vol[split:]

# -----------------------------
# 7️⃣ Build LSTM models
# -----------------------------
def build_lstm(input_shape, output_dim):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model_price = build_lstm((SEQ_LEN, X_train_price.shape[2]), len(target_cols_price))
model_volume = build_lstm((SEQ_LEN, X_train_vol.shape[2]), len(target_cols_volume))

# -----------------------------
# 8️⃣ Train models
# -----------------------------
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model_price.fit(X_train_price, y_train_price, validation_data=(X_test_price, y_test_price),
                epochs=100, batch_size=16, callbacks=[es], verbose=1)
model_volume.fit(X_train_vol, y_train_vol, validation_data=(X_test_vol, y_test_vol),
                 epochs=100, batch_size=16, callbacks=[es], verbose=1)

# -----------------------------
# 9️⃣ Predictions & metrics
# -----------------------------
def evaluate_model(model, X_test, y_test, scaler, target_names):
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test)
    
    print("\n📊 Per-Feature Evaluation Metrics:")
    for i, col in enumerate(target_names):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        print(f"{col:<15} → RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")
    return y_true, y_pred

y_true_price, y_pred_price = evaluate_model(model_price, X_test_price, y_test_price, scaler_y_price, target_cols_price)
y_true_vol, y_pred_vol = evaluate_model(model_volume, X_test_vol, y_test_vol, scaler_y_vol, target_cols_volume)

# -----------------------------
# 🔟 7-Day Forecast
# -----------------------------
def forecast_7_days(model, last_seq, scaler_y, features_order):
    future_scaled = []
    last_seq_copy = last_seq.copy()
    
    for _ in range(7):
        pred_scaled = model.predict(last_seq_copy.reshape(1, SEQ_LEN, len(features_order)), verbose=0)[0]
        future_scaled.append(pred_scaled)
        
        # Prepare next input
        new_row = last_seq_copy[-1].copy()
        if len(pred_scaled) == 1:
            new_row[features_order.index('stock_volume')] = pred_scaled[0]
        else:
            # Map predicted price targets to features
            new_row[features_order.index('stock_open')] = pred_scaled[1]
            new_row[features_order.index('stock_high')] = pred_scaled[2]
            new_row[features_order.index('stock_low')] = pred_scaled[3]
        last_seq_copy = np.vstack([last_seq_copy[1:], new_row])
    
    return scaler_y.inverse_transform(np.array(future_scaled))

last_seq_price = scaled_features[-SEQ_LEN:]
last_seq_vol = scaled_features[-SEQ_LEN:]

future_price = forecast_7_days(model_price, last_seq_price, scaler_y_price, feature_cols)
future_vol = forecast_7_days(model_volume, last_seq_vol, scaler_y_vol, feature_cols)

print("\n📅 Next 7-Day Price Forecasts:")
for i, day in enumerate(future_price, 1):
    print(f"\nDay +{i}:")
    for j, col in enumerate(target_cols_price):
        print(f"   {col:<15}: {day[j]:.2f}")

print("\n📅 Next 7-Day Volume Forecasts:")
for i, day in enumerate(future_vol, 1):
    print(f"Day +{i}:   {day[0]:.0f}")
