from datetime import datetime
from backend.services.gemini_service import GeminiService

class AIInsightsService:

    def __init__(self):
        self.gemini = GeminiService()

    def aggregate_model_signals(self):
        """
        Replace these with real values from your models
        """

        churn_risk = 0.74
        engagement_drop = 14
        negative_sentiment_change = 12
        at_risk_users = 340
        risk_trend = "Increasing"

        risk_score = int(churn_risk * 100)

        if risk_score > 75:
            risk_level = "High"
        elif risk_score > 50:
            risk_level = "Moderate"
        else:
            risk_level = "Stable"

        structured_data = {
            "churn_risk_percent": risk_score,
            "engagement_drop_percent": engagement_drop,
            "negative_sentiment_change_percent": negative_sentiment_change,
            "at_risk_users": at_risk_users,
            "risk_trend": risk_trend,
            "risk_level": risk_level
        }

        return structured_data

    def generate_ai_insights(self):
        signals = self.aggregate_model_signals()

        gemini_output = self.gemini.generate_insights(signals)

        return {
            "generated_at": datetime.utcnow(),
            "risk_level": signals["risk_level"],
            "ai_health_score": 100 - signals["churn_risk_percent"],
            "structured_signals": signals,
            "executive_insights": gemini_output["raw_text"]
        }