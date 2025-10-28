# =========================
# Enhanced Hybrid Recommender Script
# =========================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, Model
import os
import mlflow
from tqdm.keras import TqdmCallback
from tqdm import tqdm

# -------------------------
# Step 1: Load and preprocess
# -------------------------
df = pd.read_csv(r"C:\Users\palya\Desktop\intellistream\intellistream-ai\docs\user_movie_interactions.csv")

# Fill missing values
df['rating'] = df['rating'].fillna(df['rating'].mean())
df['review_text'] = df['review_text'].fillna("No Review")
df['completion_status'] = df['completion_status'].fillna("Unknown")

# Encode categorical features
le_user = LabelEncoder()
df['customer_id_enc'] = le_user.fit_transform(df['customer_id'])

le_movie = LabelEncoder()
df['movie_enc'] = le_movie.fit_transform(df['Movie Name'])

le_device = LabelEncoder()
df['device_enc'] = le_device.fit_transform(df['device'])

le_time = LabelEncoder()
df['time_of_day_enc'] = le_time.fit_transform(df['time_of_day'])

le_status = LabelEncoder()
df['completion_status_enc'] = le_status.fit_transform(df['completion_status'])

# Scale numeric features
scaler = MinMaxScaler()
df[['watch_duration_percent','skipped_scenes','buffering_time_sec','rewatch_count','rating']] = scaler.fit_transform(
    df[['watch_duration_percent','skipped_scenes','buffering_time_sec','rewatch_count','rating']]
)

# Text feature: TF-IDF
tfidf = TfidfVectorizer(max_features=50)
review_features = tfidf.fit_transform(df['review_text']).toarray()

# -------------------------
# Step 2: Prepare User-Item Matrix
# -------------------------
user_item_matrix = df.pivot_table(index='customer_id_enc', columns='movie_enc', values='watch_duration_percent', fill_value=0)
user_item_matrix_np = user_item_matrix.values

# -------------------------
# Step 3: Neural Network Collaborative Filtering with Content Features
# -------------------------
num_users = df['customer_id_enc'].nunique()
num_movies = df['movie_enc'].nunique()
embedding_size = 50

# Inputs
user_input = layers.Input(shape=(1,), name='user_input')
movie_input = layers.Input(shape=(1,), name='movie_input')
device_input = layers.Input(shape=(1,), name='device_input')
time_input = layers.Input(shape=(1,), name='time_input')
status_input = layers.Input(shape=(1,), name='status_input')
review_input = layers.Input(shape=(review_features.shape[1],), name='review_input')

# Embeddings
user_emb = layers.Embedding(num_users, embedding_size)(user_input)
movie_emb = layers.Embedding(num_movies, embedding_size)(movie_input)
device_emb = layers.Embedding(df['device_enc'].nunique(), 5)(device_input)
time_emb = layers.Embedding(df['time_of_day_enc'].nunique(), 5)(time_input)
status_emb = layers.Embedding(df['completion_status_enc'].nunique(), 5)(status_input)

# Flatten embeddings
user_vec = layers.Flatten()(user_emb)
movie_vec = layers.Flatten()(movie_emb)
device_vec = layers.Flatten()(device_emb)
time_vec = layers.Flatten()(time_emb)
status_vec = layers.Flatten()(status_emb)

# Concatenate all features
concat = layers.Concatenate()([user_vec, movie_vec, device_vec, time_vec, status_vec, review_input])

# Dense layers
dense = layers.Dense(128, activation='relu')(concat)
dense = layers.Dense(64, activation='relu')(dense)
output = layers.Dense(1, activation='sigmoid')(dense)  # predict normalized watch_duration_percent

# Build and compile model
model = Model(inputs=[user_input, movie_input, device_input, time_input, status_input, review_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Prepare input arrays
X = {
    'user_input': df['customer_id_enc'].values,
    'movie_input': df['movie_enc'].values,
    'device_input': df['device_enc'].values,
    'time_input': df['time_of_day_enc'].values,
    'status_input': df['completion_status_enc'].values,
    'review_input': review_features
}
y = df['watch_duration_percent'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# Step 4: Train Model with Progress Bar
# -------------------------
os.makedirs("models", exist_ok=True)
mlflow.set_experiment("Hybrid_Recommender")

with mlflow.start_run():
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=0,
        callbacks=[TqdmCallback(verbose=1)]
    )
    
    # Save model
    model.save("models/nn_hybrid_model_with_content.h5")
    
    # Log to MLflow
    mlflow.tensorflow.log_model(tf_saved_model_dir="models/nn_hybrid_model_with_content.h5", tf_meta_graph_tags=None, tf_signature_def_key=None, artifact_path="nn_hybrid_model")
    mlflow.log_params({
        "embedding_size": embedding_size,
        "epochs": 10,
        "batch_size": 32
    })
    
    # Compute metrics
    y_pred = model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    mlflow.log_metrics({"mse": mse, "mae": mae, "rmse": rmse})
    print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# -------------------------
# Step 5: User Similarity for CF
# -------------------------
user_sim = cosine_similarity(user_item_matrix_np)

# -------------------------
# Step 6: Hybrid Recommendation Function
# -------------------------
def hybrid_recommend(user_id, top_n=5, alpha=0.5):
    user_idx = le_user.transform([user_id])[0]
    
    # NN predictions
    movies = np.arange(num_movies)
    device_input_array = np.zeros_like(movies)  # default device/time/status if unknown
    time_input_array = np.zeros_like(movies)
    status_input_array = np.zeros_like(movies)
    review_input_array = np.zeros((num_movies, review_features.shape[1]))
    
    nn_preds = model.predict(
        [np.full(num_movies, user_idx), movies, device_input_array, time_input_array, status_input_array, review_input_array],
        verbose=0
    ).flatten()
    
    # User-based CF scores
    cf_scores = user_sim[user_idx] @ user_item_matrix_np / user_sim[user_idx].sum()
    
    # Hybrid score
    hybrid_score = alpha * nn_preds + (1 - alpha) * cf_scores
    
    # Top N movie recommendations
    top_movies_idx = np.argsort(hybrid_score)[-top_n:][::-1]
    top_movies = le_movie.inverse_transform(top_movies_idx)
    
    return top_movies

# -------------------------
# Step 7: Example Recommendation
# -------------------------
user_to_recommend = 'C0001'
top_movies = hybrid_recommend(user_to_recommend, top_n=5)
print(f"Top 5 recommendations for {user_to_recommend}:")
print(top_movies)
