# =========================================
# Hybrid Recommender Route
# =========================================
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow import keras
import os

# -----------------------------
# Router Initialization
# -----------------------------
router = APIRouter(prefix="/api", tags=["Recommender"])

# -----------------------------
# Load Data and Model
# -----------------------------
DATA_PATH = r"C:\Users\palya\Desktop\intellistream\intellistream-ai\docs\user_movie_interactions.csv"
MODEL_PATH = r"C:\Users\palya\Desktop\intellistream\intellistream-ai\models\nn_hybrid_model_with_content.keras"

try:
    df = pd.read_csv(DATA_PATH)

    # Encode users & movies
    le_user = LabelEncoder()
    df['customer_id_enc'] = le_user.fit_transform(df['customer_id'])

    le_movie = LabelEncoder()
    df['movie_enc'] = le_movie.fit_transform(df['Movie Name'])

    # User-item matrix
    user_item_matrix = df.pivot_table(index='customer_id_enc', columns='movie_enc',
                                      values='watch_duration_percent', fill_value=0)
    user_item_matrix_np = user_item_matrix.values
    user_sim = cosine_similarity(user_item_matrix_np)

    # Load model
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model and data loaded successfully!")

except Exception as e:
    print("❌ Error loading model/data:", e)
    df, model, le_user, le_movie, user_sim = None, None, None, None, None

# -----------------------------
# Request Schema
# -----------------------------
class RecommendRequest(BaseModel):
    customer_id: str
    top_n: int = 5

# -----------------------------
# Hybrid Recommendation Logic
# -----------------------------
def hybrid_recommend(user_id: str, top_n=5, alpha=0.5):
    if model is None or df is None:
        raise HTTPException(status_code=500, detail="Model or data not loaded properly.")

    try:
        user_idx = le_user.transform([user_id])[0]
    except ValueError:
        return [f"User {user_id} not found in database"]

    num_movies = df['movie_enc'].nunique()

    # Dummy arrays for missing auxiliary inputs
    device_input_array = np.zeros_like(np.arange(num_movies))
    time_input_array = np.zeros_like(np.arange(num_movies))
    status_input_array = np.zeros_like(np.arange(num_movies))
    # Get the expected review_input dimension from model itself
    review_input_dim = model.input[5].shape[1]  # safely extracts the expected TF-IDF length
    review_input_array = np.zeros((num_movies, review_input_dim))

    # Neural predictions
    nn_preds = model.predict(
        [
            np.full(num_movies, user_idx),
            np.arange(num_movies),
            device_input_array,
            time_input_array,
            status_input_array,
            review_input_array
        ],
        verbose=0
    ).flatten()

    # Collaborative filtering predictions
    cf_scores = user_sim[user_idx] @ user_item_matrix_np / user_sim[user_idx].sum()

    # Hybrid score
    hybrid_score = alpha * nn_preds + (1 - alpha) * cf_scores

    # Top N movies
    top_movies_idx = np.argsort(hybrid_score)[-top_n:][::-1]
    top_movies = le_movie.inverse_transform(top_movies_idx)

    return top_movies.tolist()

# -----------------------------
# API Route
# -----------------------------
@router.post("/recommend", response_model=dict)
def recommend_movies(request: RecommendRequest):
    recommendations = hybrid_recommend(request.customer_id, top_n=request.top_n)
    return {
        "customer_id": request.customer_id,
        "recommendations": recommendations
    }

# -----------------------------
# Health Route
# -----------------------------
@router.get("/status")
def check_status():
    return {"status": "ok", "message": "Recommender route active ✅"}
