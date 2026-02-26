<<<<<<< HEAD
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ API Key not found!")
else:
    genai.configure(api_key=api_key)
    print("🔍 Checking available models...")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"✅ Found: {m.name}")
    except Exception as e:
=======
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ API Key not found!")
else:
    genai.configure(api_key=api_key)
    print("🔍 Checking available models...")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"✅ Found: {m.name}")
    except Exception as e:
>>>>>>> f0ed5dfc20981dba285e7ebe360a4e40198e5d65
        print(f"❌ Error: {e}")