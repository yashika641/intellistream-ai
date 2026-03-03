"""
scheduler_service.py

Runs background stock price updates every minute using APScheduler.
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
import requests
import logging
import threading

# ==========================================
# Configuration
# ==========================================

STOCK_SYMBOLS = ["NFLX", "DIS", "WBD", "PARA"]
STOCK_API_URL = "https://your-stock-api.com/price"  # Replace with real API
API_KEY = "YOUR_API_KEY"

# In-memory store (Replace with DB in production)
stock_cache = {}

# ==========================================
# Logger Setup
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

# Lock for thread safety
lock = threading.Lock()


# ==========================================
# Utility Functions
# ==========================================

def fetch_stock_price(symbol: str):
    """
    Fetch latest stock price from external API.
    Replace with your real provider (AlphaVantage, Finnhub, etc.)
    """
    try:
        response = requests.get(
            f"{STOCK_API_URL}?symbol={symbol}&apikey={API_KEY}",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        # Adjust according to API response structure
        return float(data["price"])

    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {e}")
        return None


def calculate_change(current: float, previous: float):
    """
    Calculate absolute and percentage change.
    """
    if previous == 0:
        return 0.0, 0.0

    absolute_change = current - previous
    percent_change = (absolute_change / previous) * 100
    return round(absolute_change, 2), round(percent_change, 2)


# ==========================================
# Main Scheduled Job
# ==========================================

def update_stock_prices():
    """
    This function runs every 1 minute.
    Fetches latest prices and updates cache/database.
    """
    logger.info("Running scheduled stock update...")

    for symbol in STOCK_SYMBOLS:
        new_price = fetch_stock_price(symbol)

        if new_price is None:
            continue

        with lock:
            previous_price = stock_cache.get(symbol, {}).get("price", new_price)

            change, percent = calculate_change(new_price, previous_price)

            stock_cache[symbol] = {
                "symbol": symbol,
                "price": new_price,
                "change": change,
                "percent_change": percent,
                "last_updated": datetime.utcnow().isoformat()
            }

            logger.info(
                f"{symbol} updated | Price: {new_price} | "
                f"Change: {change} ({percent}%)"
            )


# ==========================================
# Scheduler Setup
# ==========================================

scheduler = BackgroundScheduler()

def start_scheduler():
    """
    Start APScheduler.
    """
    scheduler.add_job(
        update_stock_prices,
        trigger=IntervalTrigger(minutes=1),
        id="stock_price_job",
        name="Update stock prices every minute",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Scheduler started (runs every 1 minute)")


def shutdown_scheduler():
    """
    Shutdown scheduler gracefully.
    """
    scheduler.shutdown()
    logger.info("Scheduler stopped")