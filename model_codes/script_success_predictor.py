# ======================================
# 🧠 Script Success Predictor - Full Pipeline
# ======================================

import os
import glob
import warnings
warnings.filterwarnings("ignore")

import mlflow
import mlflow.tensorflow

import numpy as np
import pandas as pd
import joblib

from tqdm import tqdm
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import tensorflow as tf
from tensorflow import keras #type: ignore
from tensorflow.keras import layers #type: ignore


# ======================================
# 📁 PATH SETUP
# ======================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "docs", "movie_metadata.csv")
SCRIPT_PATH = os.path.join(BASE_DIR, "docs", "scripts")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

mlflow.set_experiment("Script_Success_Predictor_Final")


# ======================================
# 🧱 LOAD DATA
# ======================================

print("📥 Loading metadata...")
data = pd.read_csv(DATA_PATH)

print("📥 Loading scripts...")
scripts_data = []
files = glob.glob(os.path.join(SCRIPT_PATH, "*.txt"))

for file in tqdm(files, desc="Loading Scripts"):
    name = os.path.basename(file).replace(".txt", "").replace("_", " ")
    try:
        with open(file, "r", encoding="utf-8") as f:
            scripts_data.append({
                "Movie_Name": name.strip(),
                "Script_Text": f.read()
            })
    except Exception as e:
        print(f"⚠️ Error reading {file}: {e}")

scripts_df = pd.DataFrame(scripts_data)

data = pd.merge(data, scripts_df, on="Movie_Name", how="inner")

print(f"✅ Dataset shape after merge: {data.shape}")


# ======================================
# 🎯 TARGET CREATION
# ======================================

data["Success"] = data["IMDb_Rating"].apply(lambda x: 1 if x >= 6.0 else 0)


# ======================================
# 🧹 NUMERIC CLEANING
# ======================================

for col in ["Release_Year", "Duration"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

num_imputer = SimpleImputer(strategy="median")
data[["Release_Year", "Duration"]] = num_imputer.fit_transform(
    data[["Release_Year", "Duration"]]
)


# ======================================
# 🎬 FEATURE ENGINEERING
# ======================================

# Multi-hot encode Genres
all_genres = set()
for g_list in data["Genre"].dropna():
    all_genres.update([g.strip() for g in g_list.split(",")])

for g in all_genres:
    data[f"Genre_{g}"] = data["Genre"].apply(
        lambda x: int(g in x) if pd.notna(x) else 0
    )

data["Num_Genres"] = data["Genre"].apply(
    lambda x: len(x.split(",")) if pd.notna(x) else 0
)

data["Decade"] = (data["Release_Year"] // 10) * 10
data["Country_of_Origin"] = data["Country_of_Origin"].fillna("Unknown")


# ======================================
# 📝 TEXT FEATURES
# ======================================

print("🔍 Creating TF-IDF features...")
tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
X_text = tfidf.fit_transform(data["Script_Text"])


# ======================================
# 🧮 META FEATURES
# ======================================

meta_cols = [
    "Age_Rating",
    "Release_Year",
    "Duration",
    "Num_Genres",
    "Decade",
    "Country_of_Origin",
] + [f"Genre_{g}" for g in all_genres]

categorical_cols = ["Age_Rating", "Country_of_Origin"]
numeric_cols = [
    "Release_Year",
    "Duration",
    "Num_Genres",
    "Decade",
] + [f"Genre_{g}" for g in all_genres]

ct = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numeric_cols),
])

X_meta = ct.fit_transform(data[meta_cols])

# Combine text + meta
X = hstack([X_text, X_meta])
y = data["Success"]

print(f"✅ Feature matrix shape: {X.shape}")


# ======================================
# 🤖 MODEL DEFINITION
# ======================================

def build_dense_nn(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ======================================
# 🚀 TRAINING
# ======================================

def train_model():

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()

    with mlflow.start_run(run_name="DenseNN_TFIDF_Final"):

        model = build_dense_nn(X_train_dense.shape[1])

        history = model.fit(
            X_train_dense,
            y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )

        # -----------------------
        # Evaluation
        # -----------------------
        y_pred = (model.predict(X_test_dense) > 0.5).astype(int).flatten()

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print("\n📊 MODEL PERFORMANCE")
        print("=" * 50)
        print(f"Accuracy   : {acc:.4f}")
        print(f"F1 Score   : {f1:.4f}")
        print(f"Precision  : {precision:.4f}")
        print(f"Recall     : {recall:.4f}")
        print("=" * 50)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # -----------------------
        # SAVE EVERYTHING
        # -----------------------
        # -----------------------
# SAVE SINGLE FILE
# -----------------------

        # -----------------------
# SAVE PROPERLY
# -----------------------

        # 1️⃣ Save Keras model properly
        model.save(os.path.join(MODEL_DIR, "ScriptSuccess_Model.keras"))

        # 2️⃣ Save sklearn + metadata separately
        metadata = {
            "tfidf": tfidf,
            "column_transformer": ct,
            "imputer": num_imputer,
            "genre_list": list(all_genres)
        }

        joblib.dump(metadata, os.path.join(MODEL_DIR, "ScriptSuccess_Metadata.pkl"))

        print("\n✅ Model and metadata saved successfully!")
        print("\n✅ Single model file saved successfully!")

        return model


# ======================================
# ▶️ RUN
# ======================================

if __name__ == "__main__":
    trained_model = train_model()
    print("\n🎉 Training pipeline completed successfully!")