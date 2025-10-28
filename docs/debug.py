import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
from tqdm import tqdm  # <-- progress bars

fake = Faker()
np.random.seed(42)
random.seed(42)

# -----------------------------
# 1️⃣ Load Base Movie Metadata
# -----------------------------
movies = pd.read_csv(r"C:\Users\palya\Desktop\intellistream\intellistream-ai\docs\movie_metadata.csv")
movie_names = movies["Movie Name"].tolist()

print("\n🎬 Loaded movie metadata successfully — total movies:", len(movie_names))

# -----------------------------
# 2️⃣ Generate Customer Metadata
# -----------------------------
n_customers = 250
subscription_plans = ["Free Trial", "Basic", "Standard", "Premium"]
countries = ["United States", "India", "United Kingdom", "Germany", "France", "Canada", "Australia"]
devices = ["Mobile", "Laptop", "Smart TV", "Tablet"]
payment_methods = ["Credit Card", "Debit Card", "UPI", "PayPal", "NetBanking"]

customers = []

print("\n👥 Generating customer metadata...")
for i in tqdm(range(n_customers), desc="Creating Customers", colour="cyan"):
    signup_date = fake.date_between(start_date='-3y', end_date='-3m')
    last_login = signup_date + timedelta(days=random.randint(30, 1200))
    if random.random() < 0.05:
        last_login = None  # simulate missing logins
    tenure_months = (datetime.now().year - signup_date.year) * 12 + (datetime.now().month - signup_date.month)
    
    churned = 1 if (last_login is None or (datetime.now().date() - (last_login if last_login else datetime.now().date())).days > 60) else 0
    total_watch_hours = round(random.uniform(10, 800), 2)
    avg_watch_per_week = round(total_watch_hours / max(tenure_months * 4, 1), 2)
    preferred_genre = random.choice(["Action", "Drama", "Romance", "Thriller", "Comedy", "Science Fiction", "Documentary"])
    satisfaction_score = round(random.uniform(4.0, 9.8), 2) if random.random() > 0.05 else np.nan
    total_movies_watched = random.randint(10, 500)
    complaints = random.randint(0, 3) if random.random() < 0.15 else 0
    auto_renew = random.choice([0, 1])
    
    customers.append({
        "customer_id": f"C{str(i+1).zfill(4)}",
        "name": fake.first_name(),
        "gender": random.choice(["Male", "Female", "Other"]),
        "age": random.randint(18, 70),
        "country": random.choice(countries),
        "subscription_plan": random.choice(subscription_plans),
        "signup_date": signup_date,
        "last_login": last_login,
        "total_watch_hours": total_watch_hours,
        "avg_watch_per_week": avg_watch_per_week,
        "device_type": random.choice(devices),
        "payment_method": random.choice(payment_methods),
        "tenure_months": tenure_months,
        "churned": churned,
        "preferred_genre": preferred_genre,
        "total_movies_watched": total_movies_watched,
        "satisfaction_score": satisfaction_score,
        "complaints_raised": complaints,
        "auto_renewal_enabled": auto_renew
    })

customers_df = pd.DataFrame(customers)

# -----------------------------
# 3️⃣ Generate User-Movie Interactions
# -----------------------------
print("\n🎞 Generating user-movie interactions...")
interactions = []
time_of_day_choices = ["Morning", "Afternoon", "Evening", "Night"]

for cust in tqdm(customers_df["customer_id"], desc="Creating Interactions", colour="green"):
    n_interactions = random.randint(5, 20)  # movies watched per user
    for _ in range(n_interactions):
        movie = random.choice(movie_names)
        watch_percent = random.randint(20, 100)
        liked = 1 if watch_percent > 70 else 0
        rating = round(random.uniform(4, 10), 1) if random.random() < 0.8 else np.nan
        rewatch_count = random.randint(0, 4)
        completion_status = "Completed" if watch_percent >= 90 else random.choice(["Dropped", "Partial"])
        time_of_day = random.choice(time_of_day_choices)
        skipped_scenes = random.randint(0, 10) if watch_percent < 80 else random.randint(0, 2)
        buffering_time = round(random.uniform(0, 20), 2) if random.random() > 0.3 else 0.0
        review = random.choice([
            "Loved it!", "Too slow.", "Great acting!", "Not my type.", 
            "Could be better.", "Amazing cinematography!", "Average experience.", 
            "Would watch again!"
        ]) if random.random() < 0.5 else ""
        watch_date = fake.date_between(start_date='-1y', end_date='today')

        interactions.append({
            "customer_id": cust,
            "Movie Name": movie,
            "watch_duration_percent": watch_percent,
            "liked": liked,
            "rating": rating,
            "watch_date": watch_date,
            "rewatch_count": rewatch_count,
            "completion_status": completion_status,
            "device": random.choice(devices),
            "time_of_day": time_of_day,
            "skipped_scenes": skipped_scenes,
            "buffering_time_sec": buffering_time,
            "review_text": review
        })

interactions_df = pd.DataFrame(interactions)

# Introduce small percentage of missing values
for col in ["rating", "completion_status", "device"]:
    if random.random() < 0.1:
        interactions_df.loc[interactions_df.sample(frac=0.05).index, col] = np.nan

# -----------------------------
# 4️⃣ Save to CSV
# -----------------------------
customers_df.to_csv("customer_metadata.csv", index=False)
interactions_df.to_csv("user_movie_interactions.csv", index=False)

print("\n✅ Datasets successfully generated:")
print("   • customer_metadata.csv")
print("   • user_movie_interactions.csv")
print("\n📈 Total Customers:", len(customers_df))
print("🎥 Total Interactions:", len(interactions_df))
