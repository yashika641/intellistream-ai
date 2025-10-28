# ======================================
# 🧠 BERTDense + Meta + Scripts (One-Hot Genre) with Fine-Tuning & Balanced Classes
# ======================================

import os, glob
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFAutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------------------
# 1️⃣ Load Movie Metadata
# ------------------------------
data = pd.read_csv(r"C:\Users\palya\Desktop\intellistream\intellistream-ai\docs\movie_metadata.csv")
data['Success'] = (data['IMDb_Rating'] >= 6.0).astype(int)

# ------------------------------
# 2️⃣ Load Scripts
# ------------------------------
script_path = r"C:\Users\palya\Desktop\intellistream\intellistream-ai\docs\scripts"
scripts_data = []

for file in glob.glob(os.path.join(script_path, "*.txt")):
    name = os.path.basename(file).replace(".txt", "").replace("_", " ")
    try:
        with open(file, 'r', encoding='utf-8') as f:
            scripts_data.append({"Movie_Name": name.strip(), "Script_Text": f.read()})
    except Exception as e:
        print(f"⚠️ Error reading {file}: {e}")

scripts_df = pd.DataFrame(scripts_data)
data = pd.merge(data, scripts_df, on="Movie_Name", how="inner")

# ------------------------------
# 3️⃣ Feature Engineering
# ------------------------------
data['Genre'] = data['Genre'].fillna('Unknown')
data['Release_Year'] = pd.to_numeric(data['Release_Year'], errors='coerce')
data['Duration'] = pd.to_numeric(data['Duration'], errors='coerce')
data['IMDb_Rating'] = pd.to_numeric(data['IMDb_Rating'], errors='coerce')

data['Num_Genres'] = data['Genre'].apply(lambda x: len(x.split(',')) if pd.notna(x) else 0)
data['Decade'] = (data['Release_Year'] // 10) * 10

# Fill numeric missing values
numeric_cols = ['Release_Year', 'Duration', 'Num_Genres', 'Decade']
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Categorical features
categorical_cols = ['Age_Rating', 'Country_of_Origin']
data[categorical_cols] = data[categorical_cols].fillna('Unknown')

# ------------------- One-Hot Encode Genres -------------------
all_genres = sorted(set(g.strip() for s in data['Genre'].dropna() for g in s.split(',')))
for g in all_genres:
    data[f'Genre_{g}'] = data['Genre'].apply(lambda x: int(g in x) if pd.notna(x) else 0)

# Column transformer: scale numeric, one-hot categorical
ct = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])
X_meta = ct.fit_transform(data[numeric_cols + categorical_cols])
genre_cols = [f'Genre_{g}' for g in all_genres]
X_meta = np.hstack([X_meta.toarray(), data[genre_cols].values])

# Target
y = data['Success'].values

# ------------------------------
# 4️⃣ Train-Test Split
# ------------------------------
X_meta_train, X_meta_test, y_train, y_test, scripts_train, scripts_test = train_test_split(
    X_meta, y, data['Script_Text'].tolist(), test_size=0.2, stratify=y, random_state=42
)

# ------------------------------
# 5️⃣ Tokenize Scripts for BERT
# ------------------------------
bert_model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(bert_model_name, use_safetensors=True)
bert_model = TFAutoModel.from_pretrained(bert_model_name, use_safetensors=True)

max_len = 128
meta_dim = X_meta_train.shape[1]

def tokenize_texts(texts):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='tf')

train_tokens = tokenize_texts(scripts_train)
test_tokens = tokenize_texts(scripts_test)

# ------------------------------
# 6️⃣ Build BERT + DenseNN + Meta
# ------------------------------
def build_bert_dense_with_meta(max_len, meta_dim):
    # Inputs
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')
    meta_input = layers.Input(shape=(meta_dim,), name='meta_features')

    # Make BERT trainable for fine-tuning
    bert_model.trainable = True
    bert_output = bert_model(input_ids, attention_mask=attention_mask).last_hidden_state
    cls_token = bert_output[:, 0, :]
    mean_pool = tf.reduce_mean(bert_output, axis=1)
    text_embedding = layers.Concatenate()([cls_token, mean_pool])
    text_embedding = layers.Dropout(0.3)(text_embedding)

    # Meta projection
    meta_proj = layers.Dense(64, activation='relu')(meta_input)
    meta_proj = layers.Dropout(0.2)(meta_proj)

    # Merge text + meta
    x = layers.Concatenate()([text_embedding, meta_proj])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=[input_ids, attention_mask, meta_input], outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-5),
        loss='binary_crossentropy',
        metrics=[keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC(name='auc')]
    )
    return model

model = build_bert_dense_with_meta(max_len=max_len, meta_dim=meta_dim)
model.summary()

# ------------------------------
# 7️⃣ Class Weights
# ------------------------------
class_weights = dict(enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))

# ------------------------------
# 8️⃣ MLflow Logging & Training
# ------------------------------
mlflow.set_experiment("Script_Success_BERTDense_OneHotGenre_Finetune")

with mlflow.start_run(run_name="BERTDense_Meta_OneHotGenre_Finetune"):

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_auc', patience=3, mode='max', restore_best_weights=True
    )

    history = model.fit(
        x={
            'input_ids': train_tokens['input_ids'],
            'attention_mask': train_tokens['attention_mask'],
            'meta_features': X_meta_train
        },
        y=y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=16,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=1
    )

    # Predict
    y_pred = (model.predict({
        'input_ids': test_tokens['input_ids'],
        'attention_mask': test_tokens['attention_mask'],
        'meta_features': X_meta_test
    }) > 0.5).astype(int).flatten()

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.tensorflow.log_model(model, artifact_path="models/BERTDense_Meta_OneHotGenre_Finetune", input_example={
        'input_ids': train_tokens['input_ids'][:2],
        'attention_mask': train_tokens['attention_mask'][:2],
        'meta_features': X_meta_train[:2]
    })

    print(f"\n✅ Metrics: Accuracy={acc:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
