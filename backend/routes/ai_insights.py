from fastapi import APIRouter
from ai_insights_services import AIInsightsService

router = APIRouter()
ai_service = AIInsightsService()

@router.get("/ai-insights")
def get_ai_insights():
    return ai_service.generate_ai_insights()