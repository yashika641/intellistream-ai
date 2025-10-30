# ======================================
# 🧠 Script Success Predictor (ML + DL) - Full Improved
# ======================================
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.sparse import hstack
from tqdm import tqdm
import glob, os

# ======== Core ML Models ========
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ======== Deep Learning ========
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sentence_transformers import SentenceTransformer

# ======== Handle Imbalance ========
from imblearn.over_sampling import SMOTE

# ======================================
# 1️⃣ LOAD DATA
# ======================================
data = pd.read_csv(r"C:\Users\palya\Desktop\intellistream\intellistream-ai\docs\movie_metadata.csv")

# Merge with scripts
script_path = r"C:\Users\palya\Desktop\intellistream\intellistream-ai\docs\scripts"
scripts_data = []
files = glob.glob(os.path.join(script_path, "*.txt"))

for file in tqdm(files, desc="Loading Scripts", unit="file"):
    name = os.path.basename(file).replace(".txt", "").replace("_", " ")
    try:
        with open(file, 'r', encoding='utf-8') as f:
            scripts_data.append({"Movie_Name": name.strip(), "Script_Text": f.read()})
    except Exception as e:
        print(f"⚠️ Error reading {file}: {e}")

scripts_df = pd.DataFrame(scripts_data)
data = pd.merge(data, scripts_df, on="Movie_Name", how="inner")

# Target
data['Success'] = data['IMDb_Rating'].apply(lambda x: 1 if x >= 6.0 else 0)

# Numeric cleaning
from sklearn.impute import SimpleImputer
for col in ['Release_Year', 'Duration']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
num_imputer = SimpleImputer(strategy='median')
data[['Release_Year', 'Duration']] = num_imputer.fit_transform(data[['Release_Year', 'Duration']])

# ======================================
# 2️⃣ FEATURE ENGINEERING
# ======================================

# Multi-hot encode Genres
all_genres = set()
for g_list in data['Genre'].dropna():
    all_genres.update([g.strip() for g in g_list.split(',')])
for g in all_genres:
    data[f'Genre_{g}'] = data['Genre'].apply(lambda x: int(g in x) if pd.notna(x) else 0)

# Engineered features
data['Num_Genres'] = data['Genre'].apply(lambda x: len(x.split(',')) if pd.notna(x) else 0)
data['Decade'] = (data['Release_Year'] // 10) * 10
data['Country_of_Origin'] = data['Country_of_Origin'].fillna('Unknown')

# ---------------- Text Features ----------------
tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
X_text = tfidf.fit_transform(data['Script_Text'])

# ---------------- Meta Features ----------------
meta_cols = ['Age_Rating', 'Release_Year', 'Duration', 'Num_Genres', 'Decade', 'Country_of_Origin'] + [f'Genre_{g}' for g in all_genres]
categorical_cols = ['Age_Rating', 'Country_of_Origin']
numeric_cols = ['Release_Year', 'Duration', 'Num_Genres', 'Decade'] + [f'Genre_{g}' for g in all_genres]

ct = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numeric_cols)
])
X_meta = ct.fit_transform(data[meta_cols])
# Combine
data.to_csv(r"C:\Users\palya\Desktop\intellistream\intellistream-ai\docs\processed_movie_metadata1.csv", index=False)
X = hstack([X_text, X_meta])
y = data['Success']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Handle imbalance with SMOTE (except for CatBoost)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ======================================
# 3️⃣ DEFINE ML MODELS
# ======================================
# models = {
#     "LogisticRegression": LogisticRegression(max_iter=500, class_weight='balanced'),
#     "RidgeClassifier": RidgeClassifier(class_weight='balanced'),
#     "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
#     "GradientBoosting": GradientBoostingClassifier(random_state=42),
#     "AdaBoost": AdaBoostClassifier(random_state=42),
#     "ExtraTrees": ExtraTreesClassifier(random_state=42, class_weight='balanced'),
#     "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss',
#                              scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()),
#     "LightGBM": LGBMClassifier(random_state=42, class_weight='balanced'),
#     "CatBoost": CatBoostClassifier(verbose=0, random_state=42, auto_class_weights='Balanced'),
# }

# # ======================================
# # 4️⃣ TRAIN + LOG ML MODELS
# # ======================================
mlflow.set_experiment("Script_Success_Predictor_FullSuite_Improved")

# for model_name, model in tqdm(models.items(), desc="Training ML Models"):
#     with mlflow.start_run(run_name=model_name):
#         if model_name == "CatBoost":
#             model.fit(X_train, y_train)  # CatBoost handles imbalance internally
#         else:
#             model.fit(X_train_res, y_train_res)

#         y_pred = model.predict(X_test)
#         acc = accuracy_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)

#         mlflow.log_param("model_name", model_name)
#         mlflow.log_metric("accuracy", acc)
#         mlflow.log_metric("f1_score", f1)
#         mlflow.log_metric("precision", precision)
#         mlflow.log_metric("recall", recall)
#         mlflow.sklearn.log_model(model, artifact_path=f"models/{model_name}", input_example=X_train[:5].toarray())

#         print(f"{model_name}: Accuracy={acc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# ======================================
# 5️⃣ DEEP LEARNING MODEL 1: TF-IDF + Meta → Dense NN
# ======================================
def build_dense_nn(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

with mlflow.start_run(run_name="DeepLearning_DenseNN"):
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()
    nn_model = build_dense_nn(X_train_dense.shape[1])
    history = nn_model.fit(X_train_dense, y_train, validation_split=0.1, epochs=10, batch_size=32, verbose=1)
    # Save locally
    nn_model.save(r"C:\Users\palya\Desktop\intellistream\models\DenseNN_model.h5")
    print("✅ DenseNN model saved successfully!")

    y_pred_dl = (nn_model.predict(X_test_dense) > 0.5).astype(int).flatten()
    acc = accuracy_score(y_test, y_pred_dl)
    f1 = f1_score(y_test, y_pred_dl)
    precision = precision_score(y_test, y_pred_dl)
    recall = recall_score(y_test, y_pred_dl)
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.tensorflow.log_model(nn_model, artifact_path="models/DenseNN", input_example=X_train[:5].toarray())
    
    print(f"DeepLearning_DenseNN: Accuracy={acc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# ======================================
# 6️⃣ DEEP LEARNING MODEL 2: SentenceTransformer + Meta → Dense NN
# ======================================
print("\n🔍 Generating BERT embeddings...")
sbert = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = sbert.encode(data['Script_Text'].tolist(), show_progress_bar=True)

# Ensure X_meta is dense
if not isinstance(X_meta, np.ndarray):
    X_meta_dense = X_meta.toarray()
else:
    X_meta_dense = X_meta

# Concatenate embeddings + meta features
X_embed = np.hstack([embeddings, X_meta_dense])

# Train-test split
X_embed_train, X_embed_test, y_embed_train, y_embed_test = train_test_split(
    X_embed, y, test_size=0.2, stratify=y, random_state=42
)


with mlflow.start_run(run_name="DeepLearning_BERTDense"):
    bert_model = build_dense_nn(X_embed_train.shape[1])
    history = bert_model.fit(X_embed_train, y_embed_train, validation_split=0.1, epochs=10, batch_size=32, verbose=1)
    # Save locally
    # Save locally
    bert_model.save(r"C:\Users\palya\Desktop\intellistream\models\BERTDense_model.h5")
    print("✅ BERTDense model saved successfully!")


    y_pred_bert = (bert_model.predict(X_embed_test) > 0.5).astype(int).flatten()
    acc = accuracy_score(y_embed_test, y_pred_bert)
    f1 = f1_score(y_embed_test, y_pred_bert)
    precision = precision_score(y_embed_test, y_pred_bert)
    recall = recall_score(y_embed_test, y_pred_bert)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Correct input example for BERT + meta
    mlflow.tensorflow.log_model(
        bert_model, 
        artifact_path="models/BERTDense", 
        input_example=X_embed_train[:5]
    )

    print(f"DeepLearning_BERTDense: Accuracy={acc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

print("\n✅ All ML and DL models trained and logged successfully!")
