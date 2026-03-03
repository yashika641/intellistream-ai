# =============================================
# Hybrid Multi-Output Movie Prediction Model
# Improved with BatchNorm, Focal Loss, and Class-Balanced Training
# Predicts: Age Rating, Duration, IMDb Rating, Sentiment, Genre
# =============================================
from tensorflow.keras.metrics import BinaryAccuracy, AUC #type: ignore

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization, LeakyReLU #type: ignore
from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau #type: ignore
import tensorflow as tf
from tools import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ----------------------------
# 1️⃣ Load Dataset
# ----------------------------
df = pd.read_csv(os.path.join(BASE_DIR, "docs", "processed_movie_metadata.csv"))
df['Age_Rating'] = df['Age_Rating'].fillna('Unknown')
df['Duration'] = df['Duration'].fillna(df['Duration'].median())
df['IMDb_Rating'] = df['IMDb_Rating'].fillna(df['IMDb_Rating'].median())
df['Script_Text'] = df['Script_Text'].fillna('')

# ----------------------------
# 2️⃣ Encode Structured Features
# ----------------------------
le_country = LabelEncoder()
df['Country_of_Origin_enc'] = le_country.fit_transform(df['Country_of_Origin'])

le_content = LabelEncoder()
df['Content_Type_enc'] = le_content.fit_transform(df['Content_Type'])

structured_features = ['Release_Year','Country_of_Origin_enc','Num_Genres','Decade','Content_Type_enc']
X_struct = df[structured_features].values
scaler_struct = StandardScaler()
X_struct = scaler_struct.fit_transform(X_struct)

# ----------------------------
# 3️⃣ TF-IDF Text Features
# ----------------------------
tfidf = TfidfVectorizer(max_features=1000)
X_text = tfidf.fit_transform(df['Script_Text']).toarray()

# ----------------------------
# 4️⃣ Targets
# ----------------------------
# Age Rating
le_age = LabelEncoder()
y_age = le_age.fit_transform(df['Age_Rating'])

# Duration
scaler_duration = MinMaxScaler()
y_duration = scaler_duration.fit_transform(df['Duration'].values.reshape(-1,1))

# IMDb Rating
scaler_imdb = MinMaxScaler()
y_imdb = scaler_imdb.fit_transform(df['IMDb_Rating'].values.reshape(-1,1))

# Sentiment
y_sentiment = np.array([2 if r>=7 else 1 if r>=4 else 0 for r in df['IMDb_Rating']])

# Genre
genre_cols = [c for c in df.columns if c.startswith('Genre_')]
y_genre = df[genre_cols].values

# ----------------------------
# 5️⃣ Compute sample weights
# ----------------------------
age_weights = compute_class_weight('balanced', classes=np.unique(y_age), y=y_age)
age_sample_weight = np.array([age_weights[i] for i in y_age])

sentiment_weights = compute_class_weight('balanced', classes=np.unique(y_sentiment), y=y_sentiment)
sentiment_sample_weight = np.array([sentiment_weights[i] for i in y_sentiment])

# ----------------------------
# 6️⃣ Train-Test Split
# ----------------------------
X_train_struct, X_val_struct, X_train_text, X_val_text, y_train_age, y_val_age, \
y_train_duration, y_val_duration, y_train_imdb, y_val_imdb, \
y_train_sentiment, y_val_sentiment, y_train_genre, y_val_genre, \
age_sample_train, age_sample_val, sentiment_sample_train, sentiment_sample_val = train_test_split(
    X_struct, X_text, y_age, y_duration, y_imdb, y_sentiment, y_genre,
    age_sample_weight, sentiment_sample_weight,
    test_size=0.2, random_state=42
)

# ----------------------------
# 7️⃣ Focal Loss for Genre
# ----------------------------
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.exp(-bce)
        loss = alpha * (1-pt)**gamma * bce
        return tf.reduce_mean(loss)
    return focal_loss_fixed

# ----------------------------
# 8️⃣ Build Hybrid Model
# ----------------------------
# Structured branch
struct_input = Input(shape=(X_train_struct.shape[1],), name='structured_input')
x_struct = Dense(256)(struct_input)
x_struct = BatchNormalization()(x_struct)
x_struct = LeakyReLU()(x_struct)
x_struct = Dropout(0.3)(x_struct)
x_struct = Dense(128)(x_struct)
x_struct = BatchNormalization()(x_struct)
x_struct = LeakyReLU()(x_struct)

# Text branch
text_input = Input(shape=(X_train_text.shape[1],), name='text_input')
x_text = Dense(512)(text_input)
x_text = BatchNormalization()(x_text)
x_text = LeakyReLU()(x_text)
x_text = Dropout(0.3)(x_text)
x_text = Dense(256)(x_text)
x_text = BatchNormalization()(x_text)
x_text = LeakyReLU()(x_text)

# Combined
x = Concatenate()([x_struct, x_text])
x = Dense(256)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(0.3)(x)
x = Dense(128)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dense(64)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# Outputs
age_output = Dense(len(le_age.classes_), activation='softmax', name='age_rating')(x)
duration_output = Dense(1, activation='linear', name='duration')(x)
imdb_output = Dense(1, activation='linear', name='imdb_rating')(x)
sentiment_output = Dense(3, activation='softmax', name='sentiment')(x)
genre_output = Dense(y_genre.shape[1], activation='sigmoid', name='genre')(x)

model = Model(inputs=[struct_input, text_input],
              outputs=[age_output, duration_output, imdb_output, sentiment_output, genre_output])

model.compile(
    optimizer='adam',
    loss={
        'age_rating':'sparse_categorical_crossentropy',
        'duration':'mse',
        'imdb_rating':'mse',
        'sentiment':'sparse_categorical_crossentropy',
        'genre':focal_loss()
    },
    metrics={
        'age_rating':'accuracy',
        'duration':'mae',
        'imdb_rating':'mae',
        'sentiment':'accuracy',
        'genre':[BinaryAccuracy(name='binary_accuracy'), AUC(name='auc')]
    }
)

model.summary()

# ----------------------------
# 9️⃣ Callbacks
# ----------------------------
checkpoint = ModelCheckpoint('hybrid_movie_model_best.h5', save_best_only=True, monitor='val_loss', verbose=1)
earlystop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)

# ----------------------------
# 🔟 Sample Weights
# ----------------------------
sample_weight = {
    'age_rating': age_sample_train,
    'duration': np.ones_like(y_train_duration.flatten()),
    'imdb_rating': np.ones_like(y_train_imdb.flatten()),
    'sentiment': sentiment_sample_train,
    'genre': np.ones_like(y_train_genre[:,0])
}

# ----------------------------
# 1️⃣1️⃣ Train Model
# ----------------------------
history = model.fit(
    [X_train_struct, X_train_text],
    [y_train_age, y_train_duration, y_train_imdb, y_train_sentiment, y_train_genre],
    validation_data=([X_val_struct, X_val_text],
                     [y_val_age, y_val_duration, y_val_imdb, y_val_sentiment, y_val_genre]),
    epochs=50,
    batch_size=32,
    sample_weight=sample_weight,
    callbacks=[checkpoint, earlystop, reduce_lr],
    verbose=1
)

# ----------------------------
# 1️⃣2️⃣ Sample Predictions
# ----------------------------
pred_age, pred_duration, pred_imdb, pred_sentiment, pred_genre = model.predict([X_val_struct, X_val_text])

pred_age_labels = [le_age.classes_[i] for i in pred_age.argmax(axis=1)]
sentiment_map = ['Negative','Neutral','Positive']
pred_sentiment_labels = [sentiment_map[i] for i in pred_sentiment.argmax(axis=1)]
pred_genre_labels = (pred_genre > 0.3).astype(int)

# Inverse scaling
pred_duration_actual = scaler_duration.inverse_transform(pred_duration)
pred_imdb_actual = scaler_imdb.inverse_transform(pred_imdb)

print("\nSample Predictions:")
for i in range(5):
    print(f"\n--- Movie Sample {i+1} ---")
    print("Structured Input:", X_val_struct[i])
    print("Script Sample (first 20 words):", ' '.join([tfidf.get_feature_names_out()[j] for j in np.argsort(X_val_text[i])[-20:]]))
    print(f"Predicted Age Rating: {pred_age_labels[i]}")
    print(f"Predicted Duration: {pred_duration_actual[i][0]:.1f} mins")
    print(f"Predicted IMDb Rating: {pred_imdb_actual[i][0]:.1f}")
    print(f"Predicted Sentiment: {pred_sentiment_labels[i]}")
    print(f"Predicted Genres (binary): {pred_genre_labels[i]}")
