from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import recommender
from routes import script_predictor
from routes import churn

# -------------------------------
# Create FastAPI App
# -------------------------------
app = FastAPI(title="IntelliStream AI Recommender API")

# -------------------------------
# Allow CORS for React Frontend

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Include Routes
# -------------------------------
app.include_router(recommender.router)
app.include_router(script_predictor.router)
app.include_router(churn.router)


# -------------------------------
# Root Route
# -------------------------------
@app.get("/")
def root():
    return {"message": "🚀 IntelliStream AI Backend is Running!"}

# -------------------------------
# Run using: uvicorn app:app --reload
# -------------------------------
