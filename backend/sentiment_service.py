# services/dashboard_service.py

from collections import Counter, defaultdict
from datetime import datetime
import os
from typing import List, Dict
from dotenv import load_dotenv
import requests
from transformers import pipeline

# ------------------------------------------------
# Load Environment Variables
# ------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
env_path = os.path.join(BASE_DIR, "backend/.env")

print(f"🔍 Loading environment variables from: {env_path}")
load_dotenv(env_path)

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
print(NEWS_API_KEY)
# ------------------------------------------------
# Symbols You Want To Track
# ------------------------------------------------
TRACKED_SYMBOLS = [
    "Netflix",
    "Amazon",
    "Tesla",
    "Apple",
    "Disney"
]

# ------------------------------------------------
# Load Sentiment Model ONCE (Important)
# ------------------------------------------------

# ------------------------------------------------
# Helper Functions
# ------------------------------------------------
def normalize_label(label: str) -> str:
    mapping = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    }
    return mapping.get(label, label)


def sentiment_to_score(label: str) -> int:
    if label == "positive":
        return 1
    elif label == "neutral":
        return 0
    return -1


STOPWORDS = {
    "the", "is", "in", "on", "at", "for", "to", "of",
    "and", "a", "an", "with", "by", "from", "as", "it"
}

# ------------------------------------------------
# Dashboard Service
# ------------------------------------------------
class DashboardService:
    # ------------------------------------------------
    # Lazy Load Sentiment Model
    # ------------------------------------------------
    sentiment_model = None

    @staticmethod
    def get_sentiment_model():
        if DashboardService.sentiment_model is None:
            print("🔥 Loading sentiment model (PyTorch)...")

            DashboardService.sentiment_model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment",
                framework="pt",
                device=-1
            )

            print("✅ Sentiment model loaded")

        return DashboardService.sentiment_model
    def __init__(self):
        self.model = self.get_sentiment_model()
        self.base_url = "https://newsapi.org/v2/everything"

    # ------------------------------------------------
    # Fetch News
    # ------------------------------------------------
    def fetch_news(self, query: str) -> List[Dict]:

        params = {
            "q": query,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 20,
            "apiKey": NEWS_API_KEY
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json().get("articles", [])
        except Exception as e:
            print(f"❌ News fetch failed for {query}: {e}")
            return []

    # ------------------------------------------------
    # Process Single Symbol
    # ------------------------------------------------
    def process_symbol(self, symbol: str):

        articles = self.fetch_news(symbol)

        if not articles:
            return None

        titles = [a["title"] for a in articles if a.get("title")]

        if not titles:
            return None

        sentiment_results = self.model(titles)

        # Normalize labels
        for r in sentiment_results:
            r["label"] = normalize_label(r["label"])

        # -------------------------------
        # Sentiment Distribution
        # -------------------------------
        total = len(sentiment_results)

        positive = sum(1 for r in sentiment_results if r["label"] == "positive")
        neutral  = sum(1 for r in sentiment_results if r["label"] == "neutral")
        negative = sum(1 for r in sentiment_results if r["label"] == "negative")

        distribution = {
            "positive": round((positive / total) * 100, 2),
            "neutral": round((neutral / total) * 100, 2),
            "negative": round((negative / total) * 100, 2)
        }

        # -------------------------------
        # Latest News
        # -------------------------------
        latest_news = []

        for article, sentiment in zip(articles[:5], sentiment_results[:5]):
            latest_news.append({
                "title": article.get("title"),
                "source": article.get("source", {}).get("name"),
                "published_at": article.get("publishedAt"),
                "sentiment": sentiment["label"],
                "confidence": round(sentiment["score"], 3)
            })

        # -------------------------------
        # Trending Topics (Cleaned)
        # -------------------------------
        words = []

        for article in articles:
            title = article.get("title", "").lower()
            clean_words = [
                w for w in title.split()
                if w.isalpha() and w not in STOPWORDS and len(w) > 3
            ]
            words.extend(clean_words)

        trending = [
            {"topic": word, "mentions": count}
            for word, count in Counter(words).most_common(5)
        ]

        # -------------------------------
        # Sentiment Over Time (Hourly)
        # -------------------------------
        timeline = defaultdict(list)

        for article, sentiment in zip(articles, sentiment_results):
            published = article.get("publishedAt")
            if not published:
                continue

            hour_key = published[:13]  # YYYY-MM-DDTHH
            polarity = sentiment_to_score(sentiment["label"])
            timeline[hour_key].append(polarity)

        sentiment_over_time = [
            {
                "time": hour,
                "average_score": round(sum(scores) / len(scores), 3)
            }
            for hour, scores in timeline.items()
        ]

        return {
            "symbol": symbol,
            "distribution": distribution,
            "latest_news": latest_news,
            "trending_topics": trending,
            "sentiment_over_time": sentiment_over_time
        }

    # ------------------------------------------------
    # Full Dashboard Data
    # ------------------------------------------------
    def get_dashboard_data(self):

        dashboard_data = []

        for symbol in TRACKED_SYMBOLS:
            result = self.process_symbol(symbol)
            if result:
                dashboard_data.append(result)

        return {
            "updated_at": datetime.utcnow(),
            "stocks": dashboard_data
        }