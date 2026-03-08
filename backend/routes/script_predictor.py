import os
import pandas as pd
import requests
import re

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from . import model_loaders
# ------------------------------------------------
# ROUTER SETUP
# ------------------------------------------------

router = APIRouter(
    prefix="/script-success",
    tags=["Script Success Predictor"]
)



# ------------------------------------------------
# REQUEST SCHEMA (kept as requested)
# ------------------------------------------------

class ScriptInput(BaseModel):
    title: str
    genre: str
    budget: float
    runtime: float
    script_text: str

# ------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
env_path = os.path.join(BASE_DIR, ".env")
print("env path", env_path)
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
print("tmdb key " , TMDB_API_KEY)

def clean_title(title: str):
    title = re.sub(r"\b(final|draft|v\d+|\d{4})\b", "", title, flags=re.IGNORECASE)
    title = re.sub(r"[_\-]", " ", title)
    return title.strip()


def fetch_tmdb_metadata(title: str):
    if not TMDB_API_KEY:
        raise RuntimeError("TMDB_API_KEY not configured in environment variables.")

    cleaned_title = clean_title(title)
    print("clean title",cleaned_title)
    search_url = "https://api.themoviedb.org/3/search/movie"

    response = requests.get(
        search_url,
        params={
            "api_key": TMDB_API_KEY,
            "query": cleaned_title
        },
        timeout=5
    )
    print("response",response)
    if response.status_code != 200:
        return None

    results = response.json().get("results", [])
    if not results:
        return None

    # Choose most popular match
    results.sort(key=lambda x: x.get("popularity", 0), reverse=True)
    movie = results[0]
    movie_id = movie["id"]

    details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"

    details_response = requests.get(
        details_url,
        params={"api_key": TMDB_API_KEY},
        timeout=5
    )
    print("details",details_response)
    if details_response.status_code != 200:
        return None

    details = details_response.json()

    return {
        "genre": ", ".join([g["name"] for g in details.get("genres", [])]),
        "release_year": int(details["release_date"][:4])
            if details.get("release_date") else 2000,
        "duration": details.get("runtime", 120),
        "country": details["production_countries"][0]["name"]
            if details.get("production_countries") else "Unknown",
        "age_rating": "PG-13"
    }


def preprocess_input(script_text, metadata):
    df = pd.DataFrame([{
        "Genre": metadata["genre"],
        "Release_Year": metadata["release_year"],
        "Duration": metadata["duration"],
        "Country_of_Origin": metadata["country"],
        "Age_Rating": metadata["age_rating"],
        "Script_Text": script_text
    }])

    # Numeric cleaning
    df[["Release_Year", "Duration"]] = model_loaders.imputer.transform(
        df[["Release_Year", "Duration"]]
    )

    # Feature engineering
    df["Num_Genres"] = len(metadata["genre"].split(",")) if metadata["genre"] else 0
    df["Decade"] = (df["Release_Year"] // 10) * 10

    for g in model_loaders.genre_list:
        df[f"Genre_{g}"] = int(g in metadata["genre"])

    meta_cols = [
        "Age_Rating",
        "Release_Year",
        "Duration",
        "Num_Genres",
        "Decade",
        "Country_of_Origin",
    ] + [f"Genre_{g}" for g in model_loaders.genre_list]

    X_meta = model_loaders.ct.transform(df[meta_cols])
    X_text = model_loaders.tfidf.transform(df["Script_Text"])

    from scipy.sparse import hstack
    X = hstack([X_text, X_meta])

    return X.toarray()


def generate_dashboard_metrics(probability: float):
    box_office_low = int(50 + probability * 150)
    box_office_high = box_office_low + 55

    audience_score = round(6 + probability * 4, 1)
    critic_rating = int(probability * 100 * 0.95)

    return {
        "box_office_range": f"${box_office_low}M - ${box_office_high}M",
        "audience_score": audience_score,
        "critic_rating": critic_rating
    }


def extract_key_themes(script_text: str):
    themes = {
        "Redemption & Personal Growth": ["redemption", "growth", "change"],
        "Family Dynamics": ["family", "father", "mother"],
        "Justice & Morality": ["justice", "law", "crime"],
        "Sacrifice & Heroism": ["sacrifice", "hero", "battle"]
    }

    detected = []
    lower_text = script_text.lower()

    for theme, keywords in themes.items():
        score = sum(k in lower_text for k in keywords) / len(keywords)
        if score > 0:
            detected.append({
                "theme": theme,
                "score": round(score * 100, 2)
            })

    return sorted(detected, key=lambda x: x["score"], reverse=True)

def generate_genre_classification(metadata):
    genres = [g.strip() for g in metadata["genre"].split(",") if g.strip()]

    if not genres:
        return []

    # Give primary genre higher weight
    genre_scores = []
    base_score = 90

    for i, g in enumerate(genres):
        score = base_score - (i * 12)  # decreasing weight
        genre_scores.append({
            "genre": g,
            "score": max(score, 40)
        })

    return genre_scores[:5]
# ------------------------------------------------
# MAIN PREDICTION ROUTE
# ------------------------------------------------

@router.post("/analyze")
async def analyze_script(file: UploadFile = File(...)):

    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files supported")

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    script_text = content.decode("utf-8")

    # Extract title from filename
    movie_title = file.filename.replace(".txt", "").replace("_", " ").strip()

    # Fetch TMDB metadata
    metadata = fetch_tmdb_metadata(movie_title)

    if not metadata:
        raise HTTPException(status_code=404, detail="Movie not found on TMDB")

    # Preprocess
    X = preprocess_input(script_text, metadata)

    # Predict
    if model_loaders.script_model is None:
        raise HTTPException(status_code=500, detail="ML model not loaded")
    
    prediction = model_loaders.script_model.predict(X, verbose=0)
    success_probability = float(prediction[0][0])
    success_percent = round(success_probability * 100, 2)
    
    # Classification
    if success_percent >= 75:
        classification = "High probability of commercial success"
    elif success_percent >= 50:
        classification = "Moderate commercial potential"
    else:
        classification = "Low commercial viability"

    # Derived metrics
    box_office_low = int(50 + success_probability * 150)
    box_office_high = box_office_low + 55
    audience_score = round(6 + success_probability * 4, 1)
    critic_rating = int(success_probability * 100 * 0.95)
    genre_classification = generate_genre_classification(metadata)
    print("genre classification", genre_classification)
    themes = extract_key_themes(script_text)

    return {
    "title": movie_title,
    "success_probability": success_percent,
    "classification": classification,
    "box_office_range": f"${box_office_low}M - ${box_office_high}M",
    "audience_score": audience_score,
    "critic_rating": critic_rating,
    "key_themes": themes,
    "genre_classification": genre_classification, 
    "metadata_used": metadata
}