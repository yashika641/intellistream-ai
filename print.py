import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
BERT_MODEL_PATH = os.getenv("BERT_MODEL_PATH")
HYBRID_MODEL_PATH = os.getenv("HYBRID_MODEL_PATH")
DATASET_PATH = os.getenv("DATASET_PATH")

# Build absolute paths properly
bert_dense_model_path = os.path.join(BASE_DIR, BERT_MODEL_PATH)
hybrid_model_path = os.path.join(BASE_DIR, HYBRID_MODEL_PATH)
dataset_path = os.path.join(BASE_DIR, DATASET_PATH)

print("BERT path:", bert_dense_model_path)
print("Hybrid path:", hybrid_model_path)
print("Dataset path:", dataset_path)