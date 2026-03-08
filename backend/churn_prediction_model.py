import sys
import os 
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import sys
from sklearn.neural_network import MLPClassifier

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
# Base Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Gradient Boosting Family
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Tree-based Boosting Libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Support Vector Machines
from sklearn.svm import SVC
import mlflow
import mlflow.sklearn
from utils.logger import get_logger
from utils.file_handler import load_csv

logger = get_logger("churn_model_training")
from dotenv import load_dotenv
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), ".")
)
env_path = os.path.join(BASE_DIR, ".env")
if not os.path.exists(env_path):    
    raise RuntimeError(f".env file not found at {env_path}")
load_dotenv(env_path)
print("BASE_DIR from .env:", env_path)
# -----------------------------------------------------
# 🧩 PREPROCESSING
# -----------------------------------------------------
def preprocess_data(df):
    try:
        logger.info("Starting preprocessing...")

        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        df.drop(['customer_id', 'name'], axis=1, inplace=True)
        # Handle missing values
        df = df.dropna(subset=['signup_date', 'gender', 'country', 'subscription_plan'])
        df.fillna(df.median(numeric_only=True), inplace=True)
        print(df.isna().sum())
        df.fillna('Unknown', inplace=True)
        # Label encode categorical columns
        cat_cols = ['gender', 'country', 'subscription_plan', 'device_type', 'payment_method', 'preferred_genre']
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        # Convert churned to binary
        df['churned'] = df['churned'].astype(int)

        # Scale numeric features
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

        logger.info("Preprocessing complete.")
        return df

    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise

# -----------------------------------------------------
# 🧮 FEATURE ENGINEERING
# -----------------------------------------------------
def feature_engineering(df):
    try:
        logger.info("Starting feature engineering...")
        
        df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
        df['last_login'] = pd.to_datetime(df['last_login'], errors='coerce')

        # Date feature extraction
        df['signup_year'] = df['signup_date'].dt.year
        df['signup_month'] = df['signup_date'].dt.month
        df['signup_day'] = df['signup_date'].dt.day

        df['last_login_year'] = df['last_login'].dt.year
        df['last_login_month'] = df['last_login'].dt.month
        df['last_login_day'] = df['last_login'].dt.day

        # Calculate time differences
        # df['days_since_signup'] = (datetime.now() - df['signup_date']).dt.days
        # df['days_since_last_login'] = (datetime.now() - df['last_login']).dt.days

        # Fill missing last_login values safely (for users who never logged in)
        df['last_login_year'].fillna(df['signup_year'], inplace=True)
        df['last_login_month'].fillna(df['signup_month'], inplace=True)
        df['last_login_day'].fillna(df['signup_day'], inplace=True)
        # df['days_since_last_login'].fillna(df['days_since_signup'], inplace=True)

        # Behavioral ratios
        df['hours_per_movie'] = df['total_watch_hours'] / (df['total_movies_watched'] + 1e-5)
        df['engagement_ratio'] = df['avg_watch_per_week'] / (df['tenure_months'] + 1e-5)
        df['complaint_rate'] = df['complaints_raised'] / (df['tenure_months'] + 1e-5)

        # Drop original date columns
        df.drop(['signup_date', 'last_login'], axis=1, inplace=True)

        logger.info("Feature engineering completed successfully.")
        return df

    except Exception as e:
        logger.error(f"Feature engineering error: {e}")
        raise

# -----------------------------------------------------
# 🧠 MODEL TRAINING + MLFLOW LOGGING
# -----------------------------------------------------
def train_and_evaluate_models(x_train, y_train, x_test, y_test):
    try:
        logger.info("Starting model training & evaluation...")
        mlflow.set_experiment("Customer Churn Prediction")

        models = {
            # Baseline classical models
            "Logistic Regression": LogisticRegression(max_iter=5000, solver='lbfgs', class_weight='balanced'),

            # 🔥 Extra Trees (Robust alternative to Random Forest)
            "Extra Trees": ExtraTreesClassifier(n_estimators=300, class_weight='balanced', random_state=42),
        }


        report = []

        for name, model in models.items():
            with mlflow.start_run(run_name=name):
                print(f"\n🚀 Training {name}...")

                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                y_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
                import pickle
                # Save the model to a file
                pickle.dump(model, open(f'model_codes/{name}_churn_model.pkl', 'wb'))
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                roc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0

                cm = confusion_matrix(y_test, y_pred)

                # Cross-validation (for robustness)
                cv_acc = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy').mean()

                # Log MLflow metrics
                mlflow.log_params(model.get_params())
                mlflow.log_metrics({
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1,
                    "roc_auc": roc,
                    "cv_accuracy": cv_acc
                })

                mlflow.sklearn.log_model(model, artifact_path="model")

                print(f"✅ {name} | Acc: {acc:.3f}, F1: {f1:.3f}, ROC: {roc:.3f}, CV Acc: {cv_acc:.3f}")

                report.append({
                    "Model": name,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1 Score": f1,
                    "ROC AUC": roc,
                    "CV Accuracy": cv_acc
                })

        df_report = pd.DataFrame(report)
        df_report.to_csv("model_codes/churn_model_comparison.csv", index=False)
        print("\n📊 Model comparison saved to model_codes/churn_model_comparison.csv")

        best_model_name = df_report.loc[df_report["F1 Score"].idxmax(), "Model"]
        print(f"\n🏆 Best Model: {best_model_name}")

        return df_report

    except Exception as e:
        logger.error(f"Model training error: {e}")
        raise

# -----------------------------------------------------
# 🚀 MAIN FUNCTION
# -----------------------------------------------------
def main():
    logger.info("Starting Churn Prediction Pipeline")

    df = load_csv(os.path.join(BASE_DIR, "docs", "customer_metadata.csv"))

    df = preprocess_data(df)
    df = feature_engineering(df)
    # Check correlation of each feature with target
    corr = df.corr(numeric_only=True)['churned'].sort_values(ascending=False)
    print(corr.head(10))

    x = df.drop("churned", axis=1)
    print("nan value for x",x.isna().sum())
    y = df["churned"]
    print(y.isna().sum())
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=99, stratify=y)

    results = train_and_evaluate_models(x_train, y_train, x_test, y_test)
    print(results)

    logger.info("Churn Prediction Pipeline Completed Successfully")

if __name__ == "__main__":
    main()
