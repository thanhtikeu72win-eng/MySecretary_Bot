# test_env.py
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

print(f"URL Found: {url}")
print(f"Key Found: {'Yes' if key else 'No'}")