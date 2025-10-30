import os
import time
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from GoogleNews import GoogleNews
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# -------------------------
# Config
# -------------------------
CSV_FILE = "netflix_stock_news_sentiment.csv"
STOCK_SYMBOL = "NFLX"
START_DATE = datetime.strptime("2024-11-20", "%Y-%m-%d").date()
END_DATE = datetime.strptime("2025-1-29", "%Y-%m-%d").date()
BATCH_DAYS = 10
WAIT_TIME = 10  # Seconds between batches

# Keep all headlines by default
KEEP_ALL_HEADLINES = True

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Ensure CSV exists
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=[
        'date', 'headline', 'sentiment',
        'stock_open', 'stock_close', 'stock_high', 'stock_low', 'stock_volume'
    ])
    df.to_csv(CSV_FILE, index=False)

# -------------------------
# Helper functions
# -------------------------
def fetch_stock_data(date):
    try:
        data = yf.download(STOCK_SYMBOL, start=date, end=date + timedelta(days=1), progress=False)
        if not data.empty:
            row = data.iloc[0]
            # Extract only numeric values
            return {
                'open': float(row['Open']),
                'close': float(row['Close']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'volume': int(row['Volume'])
            }
        else:
            return None
    except Exception as e:
        print(f"[Stock] Error fetching data for {date}: {e}")
        return None

def fetch_google_news(date):
    googlenews = GoogleNews(start=date, end=date)
    googlenews.search("Netflix")
    news_list = googlenews.result()
    headlines = [item['title'] for item in news_list]
    return headlines if KEEP_ALL_HEADLINES else (headlines[:1] if headlines else [])

def compute_sentiment(headline):
    score = sia.polarity_scores(headline)
    if score['compound'] >= 0.05:
        return 'positive'
    elif score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def append_to_csv(data):
    df = pd.DataFrame(data, columns=[
        'date', 'headline', 'sentiment',
        'stock_open', 'stock_close', 'stock_high', 'stock_low', 'stock_volume'
    ])
    df.to_csv(CSV_FILE, mode='a', header=False, index=False)

# -------------------------
# Main loop
# -------------------------
all_data = []
date_range = pd.date_range(START_DATE, END_DATE, freq='D')

for single_date in tqdm(date_range, desc="Fetching data"):
    date_obj = single_date.date()
    date_str = date_obj.strftime('%Y-%m-%d')
    print(f"\nFetching data for {date_str}...")

    # Fetch stock data
    stock_data = fetch_stock_data(date_obj)
    if stock_data is None:
        print(f"No stock data for {date_str}. Skipping.")
        continue
    print(f"Stock: Open={stock_data['open']}, Close={stock_data['close']}, High={stock_data['high']}, Low={stock_data['low']}, Volume={stock_data['volume']}")

    # Fetch news headlines
    headlines = fetch_google_news(date_str)
    if not headlines:
        print(f"No news headlines for {date_str}.")
        continue

    print(f"Headlines ({len(headlines)}):")
    for idx, hl in enumerate(headlines, 1):
        print(f"{idx}. {hl}")

    # Combine all headlines with stock data and sentiment
    for headline in headlines:
        sentiment = compute_sentiment(headline)
        all_data.append([
            date_str, headline, sentiment,
            stock_data['open'], stock_data['close'],
            stock_data['high'], stock_data['low'], stock_data['volume']
        ])

    # Save batch every BATCH_DAYS
    if len(all_data) >= BATCH_DAYS:
        append_to_csv(all_data)
        print(f"Saved {len(all_data)} rows to {CSV_FILE}")
        all_data = []
        time.sleep(WAIT_TIME)

# Save remaining data
if all_data:
    append_to_csv(all_data)
    print(f"Saved remaining {len(all_data)} rows to {CSV_FILE}")

print("Data fetching with sentiment complete.")
