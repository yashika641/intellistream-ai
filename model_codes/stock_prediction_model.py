import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# ----------------------------
# 1️⃣ Load CSV
# ----------------------------
df = pd.read_csv(r'C:\Users\palya\Desktop\intellistream\intellistream-ai\docs\aggregated_netflix_news_stock.csv')

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
def compute_sentiment(headlines):
    scores = [sia.polarity_scores(str(headline))['compound'] for headline in headlines.split('||')]
    return np.mean(scores) if scores else 0.0

df['sentiment_score'] = df['news_headlines'].progress_apply(compute_sentiment)
df['date'] = pd.to_datetime(df['date'])
print("Initial data:\n", df.head())
# ----------------------------
# 2️⃣ Smooth numeric columns
# ----------------------------

# ROLL_WINDOW = 3
# for col in ['stock_open','stock_close','stock_high','stock_low','stock_volume']:
#     df[col] = df[col].rolling(ROLL_WINDOW, min_periods=1).mean()
# if 'sentiment_score' in df.columns:
#     df['sentiment_score'] = df['sentiment_score'].rolling(ROLL_WINDOW, min_periods=1).mean()
# print("Data after smoothing:\n", df.head())
# print(df.shape)
# ----------------------------
# 3️⃣ Aggregate per day
# ----------------------------
daily = df.groupby('date').agg({
    'news_headlines': lambda x: list(x),
    'stock_open':'first',
    'stock_close':'last',
    'stock_high':'max',
    'stock_low':'min',
    'stock_volume':'mean',
    'sentiment_score':'mean'
}).reset_index()
daily.rename(columns={'sentiment_score':'avg_sentiment', 'news_headlines':'headlines'}, inplace=True)
print("Daily aggregated data:\n", daily.head())
print(daily.shape)
print(daily.columns.to_list())
print(daily.columns)

# ----------------------------
# 4️⃣ BERT embeddings
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')
bert_model.eval()

def get_bert_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(str(text), return_tensors='pt', truncation=True, max_length=50, padding='max_length')
        outputs = bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:,0,:].numpy()
        return cls_embedding.flatten()

def get_daily_embedding(headlines):
    embeddings = np.array([get_bert_embedding(h) for h in headlines])
    return embeddings.mean(axis=0)

# Apply BERT
daily['bert_embedding'] = daily['headlines'].progress_apply(get_daily_embedding)
bert_dim = daily['bert_embedding'][0].shape[0]
bert_df = pd.DataFrame(daily['bert_embedding'].to_list(), columns=[f'bert_emb_{i}' for i in range(bert_dim)])
daily = pd.concat([daily.drop(columns=['headlines','bert_embedding']), bert_df], axis=1)
daily.dropna(inplace=True)
print("Data with BERT embeddings:\n", daily.head())
print(daily.shape)
print(daily.columns)
# ----------------------------
# 5️⃣ Prepare features & targets
# ----------------------------
feature_cols = ['avg_sentiment'] + [f'bert_emb_{i}' for i in range(bert_dim)] + ['stock_open','stock_close','stock_high','stock_low','stock_volume']
target_cols = ['stock_close']

X_values = daily[feature_cols].values
y_values = daily[target_cols].values
print("Feature sample:\n", X_values[:2])
print("Target sample:\n", y_values[:2])
# ----------------------------
# 6️⃣ Create sequences
# ----------------------------
SEQ_LEN = 7
X_seq, y_seq = [], []
for i in range(SEQ_LEN, len(daily)):
    X_seq.append(X_values[i-SEQ_LEN:i])
    y_seq.append(y_values[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# ----------------------------
# 7️⃣ PCA to reduce feature dims
# ----------------------------
# Optional: reduce BERT + numeric features with PCA
# SEQ_LEN = 7
# pca_components_per_timestep = 50
# n_components = SEQ_LEN * pca_components_per_timestep  # must match SEQ_LEN

# pca = PCA(n_components=n_components)
# X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)
# X_seq_reduced = pca.fit_transform(X_seq_flat)
# X_seq_reduced = X_seq_reduced.reshape(X_seq_reduced.shape[0], SEQ_LEN, pca_components_per_timestep)

# print("After PCA and reshape:", X_seq_reduced.shape)  # (n_samples, SEQ_LEN, 50)

# ----------------------------
# 8️⃣ Scale targets individually
# ----------------------------
# ----------------------------
# 8️⃣ Prepare log-scaled targets for the SEQUENCED ARRAY
# ----------------------------
y_seq = np.log1p(np.array(y_seq).reshape(-1, 1))

# Train-test split (aligned, both after creating sequences and after log transform)
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# Target scaler
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Model Training
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(X_train, y_train_scaled, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
model.save('news_aware_netflix_model.keras')

# ----------------------------
# 1️⃣1️⃣ Evaluation - shapes will match
# ----------------------------
y_pred_scaled = model.predict(X_test)
y_pred_rescaled = scaler_y.inverse_transform(y_pred_scaled)
y_test_rescaled = scaler_y.inverse_transform(y_test_scaled)

# Undo log transform for both arrays
y_pred_final = np.expm1(y_pred_rescaled.ravel())
y_test_final = np.expm1(y_test_rescaled.ravel())

# Metrics
mse = mean_squared_error(y_test_final, y_pred_final)
mae = mean_absolute_error(y_test_final, y_pred_final)
r2 = r2_score(y_test_final, y_pred_final)
rmse = np.sqrt(mse)
print(f"stock_close -> MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")

plt.figure(figsize=(12,6))
plt.plot(y_test_final, label='Actual Close', color='blue')
plt.plot(y_pred_final, label='Predicted Close', color='red')
plt.title('Actual vs Predicted Stock Close Prices')
plt.xlabel('Time')
plt.ylabel('Stock Close Price')
plt.legend()
plt.show()
