# backend/routes/churn_api.py
from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

router = APIRouter(prefix="/api", tags=["churn"])

# Paths
CSV_PATH = r"C:\Users\palya\Desktop\intellistream\intellistream-ai\docs\customer_metadata.csv"
MODEL_PATH = r"C:\Users\palya\Desktop\intellistream\intellistream-ai\models\Logistic Regression_churn_model.pkl"

# Load CSV and model once at startup
df_customers = pd.read_csv(CSV_PATH)
with open(MODEL_PATH, "rb") as f:
    churn_model = pickle.load(f)

# ---------------- Preprocessing Function ----------------
def preprocess_customer(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df.drop(['customer_id', 'name','churned'], axis=1, inplace=True, errors='ignore')
    
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna('Unknown', inplace=True)

    cat_cols = ['gender', 'country', 'subscription_plan', 'device_type', 'payment_method', 'preferred_genre']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).factorize()[0]
    
    return df

# ---------------- Feature Engineering Function ----------------
def feature_engineer_customer(df):
    df = df.copy()
    
    df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
    df['last_login'] = pd.to_datetime(df['last_login'], errors='coerce')

    df['signup_year'] = df['signup_date'].dt.year
    df['signup_month'] = df['signup_date'].dt.month
    df['signup_day'] = df['signup_date'].dt.day

    df['last_login_year'] = df['last_login'].dt.year.fillna(df['signup_year'])
    df['last_login_month'] = df['last_login'].dt.month.fillna(df['signup_month'])
    df['last_login_day'] = df['last_login'].dt.day.fillna(df['signup_day'])

    df['hours_per_movie'] = df['total_watch_hours'] / (df['total_movies_watched'] + 1e-5)
    df['engagement_ratio'] = df['avg_watch_per_week'] / (df['tenure_months'] + 1e-5)
    df['complaint_rate'] = df['complaints_raised'] / (df['tenure_months'] + 1e-5)

    df.drop(['signup_date', 'last_login'], axis=1, inplace=True, errors='ignore')
    
    return df

# ---------------- API Endpoints ----------------
@router.get("/predict_churn/{customer_id}")
def predict_churn(customer_id: str):
    customer_row = df_customers[df_customers['customer_id'] == customer_id]
    if customer_row.empty:
        raise HTTPException(status_code=404, detail=f"Customer ID {customer_id} not found.")
    
    customer_df = preprocess_customer(customer_row)
    customer_df = feature_engineer_customer(customer_df)
    
    churn_prob = churn_model.predict_proba(customer_df)[:, 1][0] if hasattr(churn_model, "predict_proba") else None
    churn_pred = churn_model.predict(customer_df)[0]
    
    return {
        "customer_id": customer_id,
        "churn_probability": float(churn_prob) if churn_prob is not None else None,
        "churn_prediction": int(churn_pred)
    }

@router.get("/top_churn_customers/")
def top_churn_customers(top_n: int = 10):
    df_processed = preprocess_customer(df_customers)
    df_processed = feature_engineer_customer(df_processed)

    if hasattr(churn_model, "predict_proba"):
        churn_probs = churn_model.predict_proba(df_processed)[:, 1]
    else:
        churn_probs = churn_model.predict(df_processed)
    
    df_customers_copy = df_customers.copy()
    df_customers_copy['churn_probability'] = churn_probs
    df_top = df_customers_copy.sort_values(by='churn_probability', ascending=False).head(top_n)

    return df_top[['customer_id', 'name', 'churn_probability']].to_dict(orient='records')
