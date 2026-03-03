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
from tensorflow.keras import layers, Model #type: ignore
import os
import mlflow
from tqdm.keras import TqdmCallback
from tqdm import tqdm
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# -------------------------
# Step 1: Load and preprocess
# -------------------------
df = pd.read_csv(os.path.join(BASE_DIR, "docs", "user_movie_interactions.csv"))

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
# Split all input arrays along with y
(
    X_user_train, X_user_test,
    X_movie_train, X_movie_test,
    X_device_train, X_device_test,
    X_time_train, X_time_test,
    X_status_train, X_status_test,
    X_review_train, X_review_test,
    y_train, y_test
) = train_test_split(
    X['user_input'],
    X['movie_input'],
    X['device_input'],
    X['time_input'],
    X['status_input'],
    X['review_input'],
    y,
    test_size=0.2,
    random_state=42
)

# Reconstruct dictionaries for Keras input
X_train = {
    'user_input': X_user_train,
    'movie_input': X_movie_train,
    'device_input': X_device_train,
    'time_input': X_time_train,
    'status_input': X_status_train,
    'review_input': X_review_train
}

X_test = {
    'user_input': X_user_test,
    'movie_input': X_movie_test,
    'device_input': X_device_test,
    'time_input': X_time_test,
    'status_input': X_status_test,
    'review_input': X_review_test
}

# -------------------------
# Step 4: Train Model with Progress Bar
# -------------------------
# -------------------------
# Step 4: Train Model
# -------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32,
    verbose=1
)

print("✅ Model training complete")

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
# -------------------------
# Step 8: Evaluation Metrics for Recommender
# -------------------------
def compute_recommender_metrics(model, test_df, top_k=5):
    """
    Compute Precision@K, Recall@K, NDCG@K for the hybrid recommender.
    test_df: pandas DataFrame containing the test data
    """
    # Create a dict of true movies per user in test set
    user_item_test = test_df.groupby('customer_id')['Movie Name'].apply(list).to_dict()
    
    precisions = []
    recalls = []
    ndcgs = []

    for user_id, true_movies in tqdm(user_item_test.items(), desc="Evaluating Users"):
        recommended = hybrid_recommend(user_id, top_n=top_k)
        
        # Count relevant recommendations
        relevant_count = sum(1 for movie in recommended if movie in true_movies)
        
        # Precision@K
        precision = relevant_count / top_k
        precisions.append(precision)
        
        # Recall@K
        recall = relevant_count / len(true_movies) if len(true_movies) > 0 else 0
        recalls.append(recall)
        
        # NDCG@K
        dcg = sum((1 / np.log2(idx + 2)) if movie in true_movies else 0
                  for idx, movie in enumerate(recommended))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_movies), top_k))) if len(true_movies) > 0 else 0
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)
    
    return {
        "Precision@K": np.mean(precisions),
        "Recall@K": np.mean(recalls),
        "NDCG@K": np.mean(ndcgs)
    }

# -------------------------
# Step 9: Compute metrics
# -------------------------
test_metrics = compute_recommender_metrics(model, df.iloc[X_test['user_input']], top_k=5)

print("Recommender Evaluation Metrics:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")


# =========================
# Step 10: Save Everything Properly
# =========================

import pickle

# 1️⃣ Save neural network safely (NOT pickle)
os.makedirs("models", exist_ok=True)

model.save("C:\\Users\\palya\\Desktop\\intellistream\\intellistream-ai\\models\\hybrid_recommender_model.keras")
print("✅ Neural network model saved")

# 2️⃣ Save other objects separately
hybrid_bundle = {
    "le_user": le_user,
    "le_movie": le_movie,
    "user_sim": user_sim,
    "user_item_matrix": user_item_matrix_np,
    "review_feature_size": review_features.shape[1],
    "num_movies": num_movies
}

with open("C:\\Users\\palya\\Desktop\\intellistream\\intellistream-ai\\models\\hybrid_recommender_metadata.pkl", "wb") as f:
    pickle.dump(hybrid_bundle, f)

print("✅ Metadata saved successfully")