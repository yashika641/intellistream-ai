# backend/main.py
import os
import matplotlib

matplotlib.use("Agg")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from apscheduler.schedulers.background import BackgroundScheduler

from routes import churn, recommender ,script_predictor, stock, sentiment
from services import run_churn_batch
from routes.model_loaders import load_models , load_script_success_model


# --------------------------------------------------
# Create App
# --------------------------------------------------
app = FastAPI(
    title="IntelliStream AI Recommender API",
    docs_url=None,
    redoc_url=None
)

# --------------------------------------------------
# Scheduler (Portfolio-friendly batch automation)
# --------------------------------------------------
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(run_churn_batch, "interval", hours=24)  # Run every 24 hours
# Stock dashboard job every 5 minutes



import threading



def load_all_models():
    from routes.model_loaders import load_models, load_script_success_model
    print("Loading ML models...")
    load_models()
    load_script_success_model()
    print("All ML models loaded")

@app.on_event("startup")
def startup_event():
    print("Server started")

    # run model loading AFTER server binds port
    threading.Timer(1, load_all_models).start()

@app.on_event("shutdown")
def shutdown_event():
    if scheduler.running:
        scheduler.shutdown()
        print("🛑 Scheduler stopped")

# --------------------------------------------------
# CORS (Vite frontend)
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "https://intellistream-ai.netlify.app"
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Include Routes
# --------------------------------------------------
app.include_router(churn.router)
app.include_router(recommender.router)
app.include_router(script_predictor.router)
app.include_router(stock.router)
app.include_router(sentiment.router)
# app.include_router(ai_insights.router)

# --------------------------------------------------
# Custom Dark Docs
# --------------------------------------------------
@app.get("/docs", include_in_schema=False)
async def custom_docs():
    html = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Dark Docs",
    )
    return HTMLResponse(
        content=html.body.decode() +
        """
        <style>
            body { background-color: #121212 !important; }
            .swagger-ui { background-color: #121212 !important; color: white !important; }
            .swagger-ui .topbar { background-color: #1f1f1f !important; }
        </style>
        """,
        status_code=200,
    )

# --------------------------------------------------
# Root
# --------------------------------------------------
@app.get("/")
def root():
    return {"message": "🚀 IntelliStream AI Backend is Running!"}