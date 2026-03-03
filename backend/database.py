from supabase import create_client
import os
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".")
)
env_path = os.path.join(BASE_DIR, ".env")   
print(f"🔍 Loading environment variables from: {env_path}")
load_dotenv(env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing required environment variables: SUPABASE_URL and SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)