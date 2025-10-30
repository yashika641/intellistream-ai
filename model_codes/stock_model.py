import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Tree & linear models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Neural Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.tensorflow

# ------------------ Load Data ------------------
df = pd.read_csv(r'C:\Users\palya\Desktop\intellistream\intellistream-ai\docs\netflix_news.csv')
df.dropna(subset=['headline', 'stock_open', 'stock_close', 'stock_high', 'stock_low', 'stock_volume', 'sentiment'], inplace=True)

# ------------------ Aggregate Headlines by Date & Sentiment ------------------
df_grouped = df.groupby(['date', 'sentiment']).agg({
    'headline': lambda x: ' '.join(x),
    'stock_open': 'mean',
    'stock_close': 'mean',
    'stock_high': 'mean',
    'stock_low': 'mean',
    'stock_volume': 'mean'
}).reset_index()

# ------------------ One-Hot Encode Sentiment ------------------
encoder = OneHotEncoder(sparse_output=False)
sentiment_encoded = encoder.fit_transform(df_grouped[['sentiment']])
sentiment_cols = encoder.get_feature_names_out(['sentiment'])
df_sentiment = pd.DataFrame(sentiment_encoded, columns=sentiment_cols)

# ------------------ Features & Target ------------------
X_text = df_grouped['headline']
y = df_grouped[['stock_open', 'stock_close', 'stock_high', 'stock_low', 'stock_volume']]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X_text)

# Combine TF-IDF + Sentiment One-Hot
import scipy.sparse as sp
X_final = sp.hstack([X_vect, sp.csr_matrix(df_sentiment.values)])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# ------------------ Define Models ------------------
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=200, random_state=42),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
}

# Neural Network
def build_nn(input_dim):
    model = Sequential([
        Dense(512, input_dim=input_dim, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(5, activation='linear')  # Multi-output regression
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')
    return model

# ------------------ MLflow Experiment ------------------
mlflow.set_experiment("Stock_Prediction_with_Sentiment")

best_rmse = float("inf")
best_model_name = None
best_model = None
results = []

# ------------------ Train Tree & Linear Models ------------------
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        mlflow.sklearn.log_model(model, artifact_path=f"{name}_model")

        print(f"{name} => RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = model

# ------------------ Train Neural Network ------------------
input_dim = X_train.shape[1]
nn_model = build_nn(input_dim)
with mlflow.start_run(run_name="NeuralNetwork"):
    print("\nTraining Neural Network...")
    nn_model.fit(X_train.toarray(), y_train, epochs=50, batch_size=16, verbose=0)
    y_pred_nn = nn_model.predict(X_test.toarray())

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_nn))
    mae = mean_absolute_error(y_test, y_pred_nn)
    r2 = r2_score(y_test, y_pred_nn)
    results.append({"Model": "NeuralNetwork", "RMSE": rmse, "MAE": mae, "R2": r2})

    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    mlflow.tensorflow.log_model(nn_model, artifact_path="NeuralNetwork_model")

    print(f"NeuralNetwork => RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_model_name = "NeuralNetwork"
        best_model = nn_model

# ------------------ Summary ------------------
print("\n✅ Best Model:", best_model_name, f"with RMSE: {best_rmse:.2f}")
pd.DataFrame(results).to_csv("stock_model_comparison_sentiment.csv", index=False)

# ------------------ Prediction Function ------------------
def predict_stock(headline: str, sentiment: str):
    vect = vectorizer.transform([headline])
    sent_enc = encoder.transform([[sentiment]])
    import scipy.sparse as sp
    X_input = sp.hstack([vect, sp.csr_matrix(sent_enc)])
    
    if best_model_name == "NeuralNetwork":
        pred = best_model.predict(X_input.toarray())
    else:
        pred = best_model.predict(X_input)
    return {
        "stock_open": float(pred[0][0]),
        "stock_close": float(pred[0][1]),
        "stock_high": float(pred[0][2]),
        "stock_low": float(pred[0][3]),
        "stock_volume": float(pred[0][4])
    }

# Example
example_headline = "Netflix launches new animated series"
example_sentiment = "positive"
print("\nExample Prediction:", predict_stock(example_headline, example_sentiment))
