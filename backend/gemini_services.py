import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiService:

    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate_insights(self, structured_data: dict) -> dict:
        prompt = f"""
You are a senior AI business analyst.

Analyze the following business intelligence data and generate:

1. Executive Summary (3-4 sentences, non-technical)
2. Key Risk Drivers (bullet list)
3. Business Impact (bullet list)
4. Recommended Actions (bullet list)

Keep it concise, clear, and suitable for executives.

DATA:
{structured_data}
"""

        response = self.model.generate_content(prompt)

        return {
            "raw_text": response.text
        }