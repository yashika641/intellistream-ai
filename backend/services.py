# backend/services/churn_batch.py

import os
import sys
import pickle
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import shap
import numpy as np
from datetime import datetime
# Add parent directory to path

from database import supabase
from churn_prediction_model import (
    preprocess_data,
    feature_engineering
)

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

load_dotenv()

# --------------------------------------------------
# Load Model
# --------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), ".")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "Logistic Regression_churn_model.pkl"
)

print(f"🔍 Loading model from: {MODEL_PATH}")

# with open(MODEL_PATH, "rb") as f:
#     model = pickle.load(f)

# print("✅ Supabase batch service ready")


# --------------------------------------------------
# Batch Job
# --------------------------------------------------

def run_churn_batch():
    print("🚀 Running churn batch job...")

    try:
        print("🚀 Running churn batch job...")
        # ----------------------------
        # Load Model ONLY when needed
        # ----------------------------
        print(f"🔍 Loading model from: {MODEL_PATH}")

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        print("✅ Model loaded")
        # --------------------------------------------------
        # Fetch customers from Supabase
        # --------------------------------------------------

        response = (
            supabase
            .table("customers")
            .select("*")
            .order("last_login", desc=True)
            .limit(500)
            .execute()
        )

        data = response.data

        if not data:
            print("⚠ No users found")
            return

        df = pd.DataFrame(data)

        # --------------------------------------------------
        # Preprocess & Feature Engineering
        # --------------------------------------------------

        df = preprocess_data(df)
        df = feature_engineering(df)

        X = df.reindex(columns=model.feature_names_in_, fill_value=0)

        # --------------------------------------------------
        # Predict
        # --------------------------------------------------

        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities > 0.5).astype(int)

        churn_rate = float(predictions.mean() * 100)
        retention_score = float(100 - churn_rate)
        at_risk_users = int((probabilities > 0.7).sum())
        avg_risk = float(probabilities.mean() * 100)

        # -----------------------------
        # SHAP-Based Top Churn Drivers
        # -----------------------------

        # Ensure X is the exact dataframe used for prediction
        # (same preprocessing, same feature order)

        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        # Mean absolute SHAP value per feature
        mean_importance = np.abs(shap_values.values).mean(axis=0)

        # Create feature → importance mapping
        feature_importance = dict(zip(X.columns, mean_importance))

        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Take top 4 drivers
        top_drivers = sorted_features[:4]

        # Convert to percentage contribution
        total_importance = sum([value for _, value in top_drivers])

        driver_scores = [
            (name, round((value / total_importance) * 100, 2))
            for name, value in top_drivers
        ]
        # ----------------------------------
        # User Segmentation (ML Based)
        # ----------------------------------

        df["churn_probability"] = probabilities

        # Segment logic based on risk score
        conditions = [
            df["churn_probability"] < 0.25,
            (df["churn_probability"] >= 0.25) & (df["churn_probability"] < 0.5),
            (df["churn_probability"] >= 0.5) & (df["churn_probability"] < 0.75),
            df["churn_probability"] >= 0.75
        ]

        segment_labels = [
            "Power Users",
            "Regular Viewers",
            "Casual Users",
            "High Risk"
        ]

        df["segment"] = np.select(
    conditions,
    segment_labels,
    default="Unknown"
).astype(str)

        # Compute segment metrics
        segments = []

        for label in segment_labels:
            segment_df = df[df["segment"] == label]

            if len(segment_df) == 0:
                continue

            segment_churn_rate = segment_df["churn_probability"].mean() * 100

            segments.append({
                "segment": label,
                "users": int(len(segment_df)),
                "avg_churn_probability": round(segment_churn_rate, 2)
            })
        # -----------------------------
        # Final Metrics Payload
        # -----------------------------

        metrics_payload = {
            "created_at": datetime.utcnow().isoformat(),
            "total_users": int(len(df)),
            "predicted_churn_rate": round(churn_rate, 2),
            "retention_score": round(retention_score, 2),
            "at_risk_users": int(at_risk_users),
            "average_risk_probability": round(avg_risk, 2),

            "top_drivers": [
        {"feature": name, "impact": score}
        for name, score in driver_scores
    ],
            "segments": segments

        }

        insert_response = (
            supabase
            .table("churn_metrics")
            .insert(metrics_payload)
            .execute()
        )

        if insert_response.data:
            print("✅ Metrics inserted into Supabase successfully")
        else:
            print("❌ Failed to insert metrics:", insert_response)

        print("✅ Batch completed successfully")

    except Exception as e:
        print("❌ Batch failed:", str(e))


if __name__ == "__main__":
    run_churn_batch()