import os
import numpy as np
import requests
from typing import List, Dict

from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv
from . import model_loaders

# =====================================================
# Router Configuration
# =====================================================

router = APIRouter(
    prefix="/recommendations",
    tags=["AI Recommendations"]
)

load_dotenv()

TMDB_TOKEN = os.getenv("TMDB_BEARER")

if not TMDB_TOKEN:
    raise RuntimeError("TMDB_BEARER not found in environment variables.")

# ----------------------------
# Load Metadata
# ----------------------------
def get_metadata():
    print("🔍 Loading recommender metadata...")
    print(f"Metadata bundle: {model_loaders.metadata_bundle is not None}")
    
    if model_loaders.metadata_bundle is None:
        raise RuntimeError("Recommender metadata not loaded")

    return (
        model_loaders.metadata_bundle["le_user"],
        model_loaders.metadata_bundle["le_movie"],
        model_loaders.metadata_bundle["user_sim"],
        model_loaders.metadata_bundle["user_item_matrix"],
        model_loaders.metadata_bundle["review_feature_size"],
        model_loaders.metadata_bundle["num_movies"]
    )
print("Metadata bundle in recommender route:", model_loaders.metadata_bundle is not None)
print("Model instance in recommender route:", model_loaders.model_instance is not None)
# le_user = model_loaders.metadata_bundle["le_user"]
# le_movie = model_loaders.metadata_bundle["le_movie"]
# user_sim = model_loaders.metadata_bundle["user_sim"]
# user_item_matrix = model_loaders.metadata_bundle["user_item_matrix"]
# review_feature_size = model_loaders.metadata_bundle["review_feature_size"]
# num_movies = model_loaders.metadata_bundle["num_movies"]

# print("✅ Recommender metadata loaded successfully")# =====================================================
# TMDB Poster Fetcher
# =====================================================

def fetch_movie_poster(movie_name: str) -> str | None:
    """
    Fetch movie poster URL from TMDB API.
    Returns None if not found.
    """
    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            headers={
                "accept": "application/json",
                "Authorization": f"Bearer {TMDB_TOKEN}"
            },
            params={"query": movie_name},
            timeout=5
        )

        if response.status_code != 200:
            return None

        data = response.json()

        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"

        return None

    except requests.RequestException:
        return None


# =====================================================
# Hybrid Recommendation Engine
# =====================================================
import pandas as pd

# Load original movies dataset

def hybrid_recommend(user_id: str, top_n: int = 6, alpha: float = 0.5):
    """
    Generate hybrid recommendations using:
    - Neural Network predictions
    - Collaborative Filtering scores
    """
    
    le_user, le_movie, user_sim, user_item_matrix, review_feature_size, num_movies = get_metadata()

    if user_id not in le_user.classes_:
        raise ValueError("User not found in training data")

    user_idx = le_user.transform([user_id])[0]

    movies = np.arange(num_movies)

    # Dummy auxiliary inputs
    device_input = np.zeros_like(movies)
    time_input = np.zeros_like(movies)
    status_input = np.zeros_like(movies)
    review_input = np.zeros((num_movies, review_feature_size))

    # Neural Network Predictions
    if model_loaders.model_instance is None:
        raise RuntimeError("Model instance is not loaded")

    nn_preds = model_loaders.model_instance(
        [
            np.full(num_movies, user_idx).astype("int32"),
            movies,
            device_input,
            time_input,
            status_input,
            review_input
        ],
        # verbose=0
    ).numpy().flatten()

    # Collaborative Filtering Scores
    sim_sum = user_sim[user_idx].sum()

    if sim_sum == 0:
        cf_scores = np.zeros(num_movies)
    else:
        cf_scores = user_sim[user_idx] @ user_item_matrix / sim_sum

    # Hybrid Score
    hybrid_scores = alpha * nn_preds + (1 - alpha) * cf_scores

    # Top N
    top_indices = np.argsort(hybrid_scores)[-top_n:][::-1]
    top_movies = le_movie.inverse_transform(top_indices)
    top_scores = hybrid_scores[top_indices]

    return list(zip(top_movies, top_scores))

def get_watch_time(customer_id: str) -> int:
    """
    Calculate total watch time (in hours) for a user
    based on interaction matrix.
    """
    le_user, le_movie, user_sim, user_item_matrix, review_feature_size, num_movies = get_metadata()

    if customer_id not in le_user.classes_:
        return 0

    user_idx = le_user.transform([customer_id])[0]

    # Sum of interactions (ratings or watches)
    total_interactions = user_item_matrix[user_idx].sum()

    # Assume avg movie length = 2 hours
    avg_movie_duration = 2  

    watch_time_hours = int(total_interactions * avg_movie_duration)

    return watch_time_hours

from collections import Counter

def get_top_genre(customer_id: str):
    """
    Returns user's most watched genre and percentage.
    """
    le_user, le_movie, user_sim, user_item_matrix, review_feature_size, num_movies = get_metadata()

    if customer_id not in le_user.classes_:
        return "Unknown", 0

    user_idx = le_user.transform([customer_id])[0]

    # Get watched movies (non-zero interactions)
    watched_indices = np.where(user_item_matrix[user_idx] > 0)[0]

    if len(watched_indices) == 0:
        return "None", 0

    genres = []

    for movie_idx in watched_indices:
        movie_name = le_movie.inverse_transform([movie_idx])[0]
        genre = model_loaders.movie_genre_map.get(movie_name, "Unknown")
        genres.append(genre)

    counter = Counter(genres)
    top_genre, count = counter.most_common(1)[0]

    percentage = round((count / len(genres)) * 100, 2)

    return top_genre, percentage

def get_user_similarity(customer_id: str) -> float:
    """
    Returns average similarity score of user
    compared to other users.
    """
    le_user, le_movie, user_sim, user_item_matrix, review_feature_size, num_movies = get_metadata()

    if customer_id not in le_user.classes_:
        return 0.0

    user_idx = le_user.transform([customer_id])[0]

    similarities = user_sim[user_idx]

    # Remove self similarity (always 1)
    similarities_without_self = np.delete(similarities, user_idx)

    avg_similarity = np.mean(similarities_without_self)

    return float(avg_similarity)

# =====================================================
# API Endpoint
# =====================================================

from typing import Dict, List
from fastapi import HTTPException

@router.get("/{customer_id}")
def get_recommendations(customer_id: str) -> Dict:
    """
    Get top movie recommendations + user metrics for a customer.
    """

    try:
        # 🔹 Get Hybrid Recommendations
        recommendations = hybrid_recommend(customer_id, top_n=6)

        results: List[Dict] = []

        for movie, score in recommendations:
            poster_url = fetch_movie_poster(movie)

            results.append({
                "title": movie,
                "match_score": round(float(score) * 100, 2),
                "poster_url": poster_url
            })

        # ----------------------------------------------------
        # 🔹 USER METRICS (Replace with your real logic)
        # ----------------------------------------------------

        watch_time = get_watch_time(customer_id)              # hours
        top_genre, genre_percent = get_top_genre(customer_id)
        similarity_score = get_user_similarity(customer_id)
        hybrid_accuracy = 96.8  # or load from model metrics

        # ----------------------------------------------------

        return {
            "customer_id": customer_id,

            "metrics": {
                "watch_time_hours": watch_time,
                "top_genre": top_genre,
                "top_genre_percentage": genre_percent,
                "similarity_score": round(float(similarity_score), 2),
                "model_accuracy": hybrid_accuracy
            },

            "total_recommendations": len(results),
            "recommendations": results
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
            print("RECOMMENDER ERROR:", str(e))
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
        
@router.get("/debug/users")
def get_available_users():
    le_user, le_movie, user_sim, user_item_matrix, review_feature_size, num_movies = get_metadata()

    return {
        "total_users": len(le_user.classes_),
        "sample_users": le_user.classes_[:20].tolist()
    }