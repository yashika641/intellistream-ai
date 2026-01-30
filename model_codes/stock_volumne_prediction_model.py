import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load data
df = pd.read_csv(r"C:\Users\palya\Desktop\intellistream\intellistream-ai\docs\aggregated_netflix_news_stock.csv")
df.columns = [c.strip().lower() for c in df.columns]
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)

# Aggregate daily volume
daily = df.groupby('date').agg({'stock_volume': 'sum'}).reset_index()

# Plot stock volume
plt.figure(figsize=(12,6))
plt.plot(daily['date'], daily['stock_volume'], color='blue', linewidth=1.5)
plt.title("📈 Netflix Daily Stock Volume")
plt.xlabel("Date")
plt.ylabel("Stock Volume")
plt.grid(True)
plt.show()

df['date'] = pd.to_datetime(df['date'])

# 7-day rolling average
df['volume_7d_avg'] = df['stock_volume'].rolling(window=7).mean()

# 14-day rolling average
df['volume_14d_avg'] = df['stock_volume'].rolling(window=14).mean()

# Plot original and smoothed volumes
plt.figure(figsize=(16,6))
plt.plot(df['date'], df['stock_volume'], label='Original Volume', alpha=0.5)
plt.plot(df['date'], df['volume_7d_avg'], label='7-Day Rolling Avg', color='orange', linewidth=2)
plt.plot(df['date'], df['volume_14d_avg'], label='14-Day Rolling Avg', color='green', linewidth=2)
plt.title('Stock Volume: Original vs Smoothed')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.show()
df['volume_log'] = np.log1p(df['stock_volume'])

lags = [1, 7, 14, 21]
for lag in lags:
    df[f'lag_{lag}'] = df['volume_log'].shift(lag)

# Rolling mean & std
windows = [7, 14]
for window in windows:
    df[f'roll_mean_{window}'] = df['volume_log'].rolling(window=window).mean()
    df[f'roll_std_{window}'] = df['volume_log'].rolling(window=window).std()

df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
df = pd.get_dummies(df, columns=['day_of_week'], drop_first=True)
from transformers import BertTokenizer, BertModel
import tensorflow as tf

# # # Load pretrained BERT tokenizer and model
# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # bert_model = BertModel.from_pretrained('bert-base-uncased')

# # # Function to get BERT embedding for a single headline
# # import torch
# # def get_bert_embedding(text):
# #     with torch.no_grad():
# #         inputs = tokenizer(
# #             str(text),
# #             return_tensors='pt',
# #             truncation=True,
# #             max_length=50,
# #             padding='max_length'
# #         )
# #         outputs = bert_model(**inputs)  # <-- unpack dict
# #         cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
# #         return cls_embedding.flatten()


# # # Apply to dataframe headlines
# # df['bert_embedding'] = df['headlines'].apply(get_bert_embedding)

# # # Split embedding vector into separate columns
# # bert_dim = 768
# # embeddings = np.vstack(df['bert_embedding'].values)
# from sklearn.decomposition import PCA

# bert_embeddings_reduced = PCA(n_components=32).fit_transform(embeddings)

# for i in range(bert_dim):
#     df[f'bert_emb_{i}'] = embeddings[:, i]
    
    
# df.drop(columns=['bert_embedding'], inplace=True)
df.dropna(inplace=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['headlines'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
df['sentiment_lag_1'] = df['sentiment_score'].shift(1)  # use yesterday's sentiment
df.dropna(inplace=True)

from sklearn.preprocessing import MinMaxScaler

features = [col for col in df.columns if col not in ['date',"sentiment", 'stock_volume', 'volume_7d_avg', 'volume_14d_avg',"headlines"]]
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])
df = pd.get_dummies(df, columns=['sentiment'], drop_first=True)

SEQ_LENGTH = 21  # e.g., use last 21 days to predict next day

# Features to use
X_features = features  # all scaled features except target
y_target = 'volume_log'  # predicting log-transformed volume

# Create sequences
X, y = [], []

for i in range(SEQ_LENGTH, len(df)):
    X.append(df[X_features].iloc[i-SEQ_LENGTH:i].values)
    y.append(df[y_target].iloc[i])

X, y = np.array(X), np.array(y)

print("X shape:", X.shape)  # (samples, timesteps, features)
print("y shape:", y.shape)

# 80-20 split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Train samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))  # predicting log(volume)

model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

y_pred = model.predict(X_test)
model.save('netflix_stock_volume_model.h5')
# Convert back from log
y_test_exp = np.expm1(y_test)
y_pred_exp = np.expm1(y_pred)
# Convert log predictions back to original scale
y_test_exp = np.expm1(y_test)      # actual
y_pred_exp = np.expm1(y_pred)      # predicted

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Mean Absolute Error
mae = mean_absolute_error(y_test_exp, y_pred_exp)

# Mean Squared Error
mse = mean_squared_error(y_test_exp, y_pred_exp)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# R-squared
r2 = r2_score(y_test_exp, y_pred_exp)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")


plt.figure(figsize=(16,6))
plt.plot(df['date'].iloc[split+SEQ_LENGTH:], y_test_exp, label='Actual Volume')
plt.plot(df['date'].iloc[split+SEQ_LENGTH:], y_pred_exp, label='Predicted Volume')
plt.title('Netflix Stock Volume Prediction')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.show()
