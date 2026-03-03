# backend/routes/churn.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import APIRouter
from database import supabase

router = APIRouter(
    prefix="/churn_analytics",
    tags=["Churn Analytics"]
)

# --------------------------------------------------
# Health Check
# --------------------------------------------------

@router.get("/")
def churn_home():
    return {"status": "Churn analytics service ready"}


# --------------------------------------------------
# Get Latest Metrics
# --------------------------------------------------

@router.get("/churn_analytics")
def get_latest_metrics():

    response = (
        supabase
        .table("churn_metrics")
        .select("*")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    data = response.data

    if not data:
        return {"message": "No churn metrics available yet"}

    return data[0]


# --------------------------------------------------
# Get Historical Metrics
# --------------------------------------------------

@router.get("/history")
def get_metrics_history(limit: int = 20):

    response = (
        supabase
        .table("churn_metrics")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )

    return response.data or []