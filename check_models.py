import google.generativeai as genai

# Boss ရဲ့ API KEY ကို ဒီမှာ ထည့်ပါ
GOOGLE_API_KEY = "AIzaSyCxUqSySmUw8rttC3GGJQqdR49e9voiYsw"

genai.configure(api_key=GOOGLE_API_KEY)

print("🔍 Checking available embedding models...")
try:
    found = False
    for m in genai.list_models():
        if 'embedContent' in m.supported_generation_methods:
            print(f"✅ Available: {m.name}")
            found = True
    
    if not found:
        print("❌ No embedding models found for this API Key.")
except Exception as e:
    print(f"❌ Error: {e}")