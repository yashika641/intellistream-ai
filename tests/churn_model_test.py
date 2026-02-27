import os
import sys
import pickle
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.file_handler import load_csv
from model_codes.churn_prediction_model import preprocess_data, feature_engineering

# --------------------------------------------------
# 🔹 CONFIG
# --------------------------------------------------
print("⚙️ Setting up paths and configurations...")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "Logistic Regression_churn_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "docs", "customer_metadata.csv")

print("Model Path:", MODEL_PATH)
print("Data Path:", DATA_PATH)

# MODEL_PATH = r"\models\Logistic Regression_churn_model.pkl"  # change if needed
# DATA_PATH = r"\docs\customer_metadata.csv"

# --------------------------------------------------
# 🔹 LOAD MODEL
# --------------------------------------------------

print("📦 Loading model...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("✅ Model loaded successfully.")

# --------------------------------------------------
# 🔹 LOAD DATA
# --------------------------------------------------

print("📊 Loading data...")
df = load_csv(DATA_PATH)

# Apply same preprocessing pipeline
df = preprocess_data(df)
df = feature_engineering(df)

X = df.drop("churned", axis=1)
y = df["churned"]

# --------------------------------------------------
# 🔹 RUN PREDICTIONS
# --------------------------------------------------

print("🚀 Running predictions...")

probs = model.predict_proba(X)[:, 1]
preds = (probs > 0.5).astype(int)

# --------------------------------------------------
# 🔹 KPI CALCULATIONS
# --------------------------------------------------

churn_rate = preds.mean() * 100
expected_churn = probs.mean() * 100
at_risk_users = (probs > 0.7).sum()
retention_score = (1 - probs.mean()) * 100

print("\n📊 ===== MODEL RESULTS =====")
print(f"Total Users: {len(X)}")
print(f"Predicted Churn Rate: {churn_rate:.2f}%")
print(f"Expected Churn Risk: {expected_churn:.2f}%")
print(f"At-Risk Users (>0.7 prob): {at_risk_users}")
print(f"Retention Score: {retention_score:.2f}%")

# --------------------------------------------------
# 🔹 OPTIONAL: SAVE RESULTS
# --------------------------------------------------

df["churn_probability"] = probs
df["predicted_churn"] = preds

df.to_csv("model_codes/test_predictions_output.csv", index=False)

print("\n💾 Predictions saved to model_codes/test_predictions_output.csv")