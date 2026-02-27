from fastapi import APIRouter, UploadFile, File
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import os

router = APIRouter(prefix="/api", tags=["Script Predictor"])
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# -------------------------------
# ✅ Load Models
# -------------------------------
bert_dense_model_path = os.path.join(BASE_DIR, "models", "BERTDense_model")
from keras.layers import TFSMLayer

bert_dense_model = TFSMLayer(
    bert_dense_model_path,
    call_endpoint="serving_default"
)

hybrid_model_path = os.path.join(BASE_DIR, "models", "hybrid_movie_model_best.h5")
def focal_loss_fixed(y_true, y_pred, gamma=2., alpha=.25):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
    loss = -y_true * alpha * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred) \
           - (1 - y_true) * (1 - alpha) * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred)
    return tf.reduce_mean(loss)

hybrid_model = tf.keras.models.load_model(hybrid_model_path, custom_objects={"focal_loss_fixed": focal_loss_fixed})

# -------------------------------
# 🧠 Load SBERT for embeddings
# -------------------------------
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim embeddings

# -------------------------------
# ✅ Load dataset for meta defaults
# -------------------------------
dataset_path = os.path.join(BASE_DIR, "docs", "processed_movie_metadata1.csv")
data = pd.read_csv(dataset_path)

# Numeric columns
numeric_cols = ['Release_Year', 'Duration', 'Num_Genres', 'Decade']
numeric_defaults = {col: data[col].median() for col in numeric_cols}

# Categorical columns (take first available as default)
categorical_cols = ['Age_Rating', 'Country_of_Origin']
categorical_defaults = {col: data[col].mode()[0] for col in categorical_cols}

# Genre columns
genre_cols = [c for c in data.columns if c.startswith('Genre_')]

# -------------------------------
# 🧠 Prediction Route
# -------------------------------
@router.post("/predict-script")
async def predict_script(file: UploadFile = File(...)):
    text = (await file.read()).decode("utf-8")

    # 1️⃣ SBERT embeddings
    embedding = sbert_model.encode([text])  # shape: (1, 384)

    # 2️⃣ Build meta vector
    meta_vector = []

    # Numeric
    for col in numeric_cols:
        meta_vector.append(numeric_defaults[col])

    # Genre multi-hot (all zeros as default)
    meta_vector.extend([0] * len(genre_cols))
    # genre_cols = [c for c in data.columns if c.startswith('Genre_')]
    print(len(genre_cols))  # should print 51

    # Categorical one-hot (simplified)
    meta_vector.extend([0] * len(categorical_cols))

    meta_vector = np.array(meta_vector).reshape(1, -1)

    # 3️⃣ Concatenate embedding + meta
    X_input = np.hstack([embedding, meta_vector])  # shape: (1, 441)

    # 4️⃣ Predict Script Successsuccess_output = bert_dense_model(X_input)
    success_output = bert_dense_model(X_input)
    success_score = float(success_output.numpy()[0][0])
    success_score = float(success_output[0][0])

    # 5️⃣ Predict Hybrid Model outputs
    hybrid_output = hybrid_model.predict([X_input, X_input])
    hybrid_output = [float(np.mean(x)) for x in hybrid_output]
    pred_age, pred_duration, pred_imdb, pred_sentiment, pred_genre = hybrid_output

    return {
        "success_probability": round(success_score * 100, 2),
        "predicted_age_rating": round(pred_age, 2),
        "predicted_duration": round(pred_duration, 2),
        "predicted_imdb_rating": round(pred_imdb, 2),
        "predicted_sentiment": round(pred_sentiment, 2),
        "predicted_genre": round(pred_genre, 2)
    }
