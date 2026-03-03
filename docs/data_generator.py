import os
import csv
import json
import re
import traceback
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import pipeline
import time

# === CONFIG ===
folder_path = r"C:\Users\palya\Desktop\intellistream\intellistream-ai\docs\scripts"
output_csv = r"C:\Users\palya\Desktop\intellistream\intellistream-ai\docs\movie_metadata.csv"
MAX_WORKERS = 18
MODEL_NAME = "google/flan-t5-large"
TMDB_API_KEY = "2c5c4047c26e0fce389ca5fd5be11091"  # Replace with your key

print(f"🚀 Loading model '{MODEL_NAME}' on CPU...")
try:
    llm = pipeline("text2text-generation", model=MODEL_NAME, device=-1)
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print("❌ Failed to load model:", e)
    raise SystemExit(1)


# --- TMDb lookup ---
def fetch_tmdb_data(movie_name):
    try:
        search_url = "https://api.themoviedb.org/3/search/movie"
        search_params = {"api_key": TMDB_API_KEY, "query": movie_name}
        search_result = requests.get(search_url, params=search_params).json()

        if search_result.get("results"):
            movie = search_result["results"][0]
            movie_id = movie["id"]

            # --- Fetch main movie details ---
            detail_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
            details = requests.get(detail_url, params={"api_key": TMDB_API_KEY}).json()
            genre_names = [g["name"] for g in details.get("genres", [])]

            # --- Fetch release dates for accurate age rating ---
            release_url = f"https://api.themoviedb.org/3/movie/{movie_id}/release_dates"
            release_data = requests.get(release_url, params={"api_key": TMDB_API_KEY}).json()
            age_rating = None
            if release_data.get("results"):
                for entry in release_data["results"]:
                    if entry["iso_3166_1"] == "US":  # Get US certification
                        if entry.get("release_dates"):
                            age_rating = entry["release_dates"][0].get("certification")
                        break

            return {
                "Movie Name": movie_name,
                "Genre": ", ".join(genre_names) if genre_names else None,
                "Content Type": "Movie",
                "Release Year": details.get("release_date", "").split("-")[0] if details.get("release_date") else None,
                "Country of Origin": details.get("production_countries")[0]["name"] if details.get("production_countries") else None,
                "Duration (minutes)": details.get("runtime"),
                "Age Rating": age_rating if age_rating else "Unknown",
                "IMDb Rating": details.get("vote_average")
            }
    except Exception as e:
        print(f"⚠️ TMDb lookup failed for {movie_name}: {e}")
    return {}

# --- LLM fallback ---
def extract_metadata_llm(movie_name, script_text):
    prompt = f"""
    You are a professional movie metadata AI.

    Using the movie name "{movie_name}" and this script excerpt, extract metadata as JSON:
    {{
      "Movie Name": "{movie_name}",
      "Genre": "",
      "Content Type": "",
      "Release Year": "",
      "Country of Origin": "",
      "Duration (minutes)": "",
      "Age Rating": "",
      "IMDb Rating": ""
    }}

    Fill missing values as best as possible using world knowledge or context from the script. Return **JSON only**.
    Script Excerpt:
    {script_text[:1000]}
    """
    try:
        response = llm(prompt, max_new_tokens=400)[0]["generated_text"]
        return response.strip()
    except Exception as e:
        print(f"⚠️ LLM extraction failed for {movie_name}: {e}")
        return ""


def parse_llm_response(response_text, movie_name):
    fields = ["Movie Name", "Genre", "Content Type", "Release Year", "Country of Origin",
              "Duration (minutes)", "Age Rating", "IMDb Rating"]
    metadata = {k: None for k in fields}

    # Try JSON parse
    try:
        cleaned_text = re.sub(r"^[^{]*({.*})[^}]*$", r"\1", response_text, flags=re.S)
        parsed = json.loads(cleaned_text)
        for k in fields:
            metadata[k] = parsed.get(k) or None
    except Exception:
        # Fallback regex parsing
        for k in fields:
            match = re.search(fr'"?{k}"?\s*:\s*"?(.*?)"?(,|$)', response_text, re.IGNORECASE)
            if match:
                metadata[k] = match.group(1).strip()
    metadata["Movie Name"] = metadata["Movie Name"] or movie_name
    return metadata


def process_movie(file_name):
    try:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            script_text = f.read()
        if not script_text.strip():
            raise ValueError("Empty script")

        movie_name = os.path.splitext(file_name)[0].replace("_", " ").replace("-", " ").strip()

        # Step 1: TMDb lookup
        tmdb_data = fetch_tmdb_data(movie_name)

        # Step 2: Fill missing fields with LLM
        missing_keys = [k for k, v in tmdb_data.items() if v in [None, ""]]
        if missing_keys:
            llm_response = extract_metadata_llm(movie_name, script_text)
            llm_data = parse_llm_response(llm_response, movie_name)
            for k in missing_keys:
                tmdb_data[k] = llm_data.get(k) or "Unknown"

        # Print both sources
        print(f"\n🎬 --- Movie: {movie_name} ---")
        print("TMDb Data:", tmdb_data)
        if missing_keys:
            print("LLM Fallback Data:", {k: tmdb_data[k] for k in missing_keys})
        print("-" * 60)

        return tmdb_data

    except Exception as e:
        print(f"⚠️ Error processing {file_name}: {e}")
        traceback.print_exc()
        return None


def main():
    print("🔍 Scanning folder for scripts...")
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    if not all_files:
        print("❌ No scripts found.")
        return

    all_metadata = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_movie, f): f for f in all_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing movies"):
            result = future.result()
            if result:
                all_metadata.append(result)

    if not all_metadata:
        print("❌ No metadata extracted.")
        return

    # Save CSV
    with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(all_metadata[0].keys()))
        writer.writeheader()
        writer.writerows(all_metadata)

    print(f"\n✅ Metadata CSV saved → {output_csv}")
    print(f"📊 Total movies processed: {len(all_metadata)}")


if __name__ == "__main__":
    print("✅ Script started.")
    main()
    print("✅ Script finished.")
