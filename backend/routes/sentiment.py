# config.py

TRACKED_SYMBOLS = [
    "Netflix",
    "Amazon",
    "Tesla",
    "Apple",
    "Disney"
]

from fastapi import APIRouter
from sentiment_service import DashboardService

router = APIRouter()

dashboard_service = DashboardService()

@router.get("/dashboard")
def get_dashboard(symbol: str = "Netflix"):
    return dashboard_service.process_symbol(symbol)