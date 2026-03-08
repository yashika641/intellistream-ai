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

# routes/sentiment_dashboard.py

router = APIRouter()

def get_dashboard_service():
    return DashboardService()


@router.get("/dashboard")
def get_dashboard(symbol: str = "Netflix"):
    service = get_dashboard_service()
    return service.process_symbol(symbol)