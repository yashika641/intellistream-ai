import os
import joblib
import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

# ------------------------------------------------
# ROUTER SETUP
# ------------------------------------------------

router = APIRouter(
    prefix="/script-success",
    tags=["Script Success Predictor"]
)

# ------------------------------------------------
# LOAD SAVED PIPELINE (SINGLE FILE)
# ------------------------------------------------
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
MODEL_PATH = os.path.join(BASE_DIR, "models", "ScriptSuccess_FullPipeline.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model file not found!")

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
tfidf = bundle["tfidf"]
ct = bundle["column_transformer"]
imputer = bundle["imputer"]
genre_list = bundle["genre_list"]

# ------------------------------------------------
# REQUEST SCHEMA
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

def preprocess_input(data: ScriptInput):
    df = pd.DataFrame([{
        "genre": data.genre,
        "budget": data.budget,
        "runtime": data.runtime,
        "script_text": data.script_text
    }])

    # Numeric preprocessing
    df[["budget", "runtime"]] = imputer.transform(df[["budget", "runtime"]])

    # Categorical transform
    categorical = ct.transform(df[["genre"]])

    # TFIDF transform
    text_features = tfidf.transform(df["script_text"])

    # Combine everything
    from scipy.sparse import hstack
    X = hstack([categorical, text_features, df[["budget", "runtime"]]])

    return X


def generate_dashboard_metrics(probability: float):
    """
    Generate dashboard metrics based on success probability.
    This keeps UI dynamic.
    """

    box_office_low = int(50 + probability * 150)
    box_office_high = box_office_low + 55

    audience_score = round(6 + probability * 4, 1)
    critic_rating = int(probability * 100 * 0.95)

    return {
        "box_office_range": f"${box_office_low}M - ${box_office_high}M",
        "audience_score": audience_score,
        "critic_rating": critic_rating
    }


def extract_top_genres(probability_vector):
    genre_scores = dict(zip(genre_list, probability_vector))

    sorted_genres = sorted(
        genre_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        {"genre": g[0], "score": round(float(g[1]) * 100, 2)}
        for g in sorted_genres[:5]
    ]


def extract_key_themes(script_text: str):
    """
    Simple keyword-based theme detection.
    Replace with NLP later if needed.
    """

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


# ------------------------------------------------
# MAIN PREDICTION ROUTE
# ------------------------------------------------
@router.post("/predict")
async def predict_script(file: UploadFile = File(...)):

    # Validate file presence
    if not file:
        raise HTTPException(status_code=400, detail="File is required")

    # Validate file type
    if not file.filename.endswith(".txt"):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are supported"
        )

    try:
        # Read file
        content = await file.read()

        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        script_text = content.decode("utf-8")

        # Preprocess
        X = preprocess_input(script_text)  # IMPORTANT: must match training pipeline

        # Predict
        prediction = model.predict(X)
        success_probability = float(prediction[0][0])

        # Generate metrics
        dashboard_metrics = generate_dashboard_metrics(success_probability)
        genres = extract_top_genres(success_probability)
        themes = extract_key_themes(script_text)

        return {
            "success_probability": round(success_probability * 100, 2),
            "box_office": dashboard_metrics["box_office_range"],
            "audience_score": dashboard_metrics["audience_score"],
            "critic_rating": dashboard_metrics["critic_rating"],
            "genre_classification": genres,
            "key_themes": themes
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )