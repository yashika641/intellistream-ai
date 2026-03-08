# backend/model_loader.py
import os
import pickle
import pandas as pd
from keras.models import load_model

model_instance = None
metadata_bundle = None
movie_genre_map = None

def load_models():

    global model_instance, metadata_bundle, movie_genre_map

    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..",".")
    )

    MODEL_PATH = os.path.join(BASE_DIR, "models", "hybrid_recommender_model.keras")
    META_PATH = os.path.join(BASE_DIR, "models", "hybrid_recommender_metadata.pkl")

    print("🔍 Loading recommender model...")

    model_instance = load_model(MODEL_PATH)
    print(MODEL_PATH)
    with open(META_PATH, "rb") as f:
        metadata_bundle = pickle.load(f)

    movies_df = pd.read_csv(os.path.join(BASE_DIR, "movie_metadata.csv"))
    movie_genre_map = dict(zip(movies_df["Movie_Name"], movies_df["Genre"]))

    print("✅ All ML models loaded successfully")
    print(metadata_bundle.keys())
    
    
    
import os
import joblib
from keras.models import load_model

script_model = None
script_metadata = None

tfidf = None
ct = None
imputer = None
genre_list = None


def load_script_success_model():

    global script_model
    global script_metadata
    global tfidf, ct, imputer, genre_list

    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..",".")
    )

    MODEL_PATH = os.path.join(BASE_DIR, "models", "ScriptSuccess_Model.keras")
    META_PATH = os.path.join(BASE_DIR, "models", "ScriptSuccess_Metadata.pkl")

    print("🔍 Loading Script Success model...")

    script_model = load_model(MODEL_PATH)

    script_metadata = joblib.load(META_PATH)

    tfidf = script_metadata["tfidf"]
    ct = script_metadata["column_transformer"]
    imputer = script_metadata["imputer"]
    genre_list = script_metadata["genre_list"]

    print("✅ Script Success model loaded successfully")