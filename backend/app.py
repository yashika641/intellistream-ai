# backend/main.py
import os

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
from routes.stock import run_stock_dashboard_batch


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
scheduler.add_job(
    run_stock_dashboard_batch,
    "interval",
    minutes=5
)

import threading

@app.on_event("startup")
def startup_event():
    if not scheduler.running:
        scheduler.start()
        print("✅ Scheduler started")

    # Load ML models in background so server starts instantly
    threading.Thread(target=load_models).start()
    threading.Thread(target=load_script_success_model).start()
    

@app.on_event("shutdown")
def shutdown_event():
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