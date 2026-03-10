import os
import telebot
import requests
import json
import google.generativeai as genai
from datetime import datetime

# 2. Load Env Vars & n8n Config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mysecretary79-bot")

# n8n Production Webhook URL
N8N_WEBHOOK_URL = "https://thanhtike72win-n8n-server.hf.space/webhook/add-expense"

# Initialize Bot and Gemini
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_text = message.text
    # ယနေ့ ရက်စွဲကို ယူခြင်း
    today_date = datetime.now().strftime("%Y-%m-%d")

    # AI ကို ညွှန်ကြားချက် ပေးခြင်း (Prompt)
    prompt = f"""
    You are an expense tracker assistant. Extract the Date, Description, and Amount from the following Burmese text.
    Today's date is: {today_date}
    User Text: "{user_text}"

    Rules:
    - Date must be in YYYY-MM-DD format. If the user doesn't specify a date or says "ဒီနေ့" (today), use {today_date}. If they say "မနေ့က" (yesterday) etc., calculate accordingly.
    - Description should be a short, clear summary (e.g., "Coffee", "Taxi", "Lunch"). You can translate it to English or keep it as a short Burmese word.
    - Amount must be a number only (e.g., 5000). Remove any currency text like "ကျပ်" or "ks".

    Output ONLY a raw JSON object, nothing else. Do not use Markdown blocks.
    Example:
    {{"Date": "2024-03-26", "Description": "Coffee", "Amount": "5000"}}
    """

    try:
        # ၁။ AI (Gemini) ဆီမှ Data ကို ခွဲထုတ်ခိုင်းခြင်း
        response = model.generate_content(prompt)
        
        # AI ပြန်ပေးတဲ့ စာသားထဲက JSON ကို သန့်စင်ခြင်း
        json_string = response.text.strip().replace("```json", "").replace("```", "")
        extracted_data = json.loads(json_string)

        print("AI Extracted Data:", extracted_data) # Log တွင် စစ်ဆေးရန်

        # ၂။ ခွဲထုတ်ရရှိတဲ့ Data ကို n8n (Google Sheets) ဆီသို့ Webhook မှတစ်ဆင့် ပို့ခြင်း
        headers = {'Content-Type': 'application/json'}
        n8n_response = requests.post(N8N_WEBHOOK_URL, json=extracted_data, headers=headers)

        # ၃။ Telegram တွင် User ကို ပြန်အကြောင်းကြားခြင်း
        if n8n_response.status_code == 200:
            bot_reply = f"✅ စာရင်းသွင်းပြီးပါပြီ!\n\n📅 Date: {extracted_data['Date']}\n📝 Item: {extracted_data['Description']}\n💰 Amount: {extracted_data['Amount']} ks"
            bot.reply_to(message, bot_reply)
        else:
            bot.reply_to(message, f"❌ Google Sheets ဆီသို့ ပို့ရာတွင် အဆင်မပြေဖြစ်နေပါသည်။ (Error: {n8n_response.status_code})")

    except Exception as e:
        print("Error:", e)
        bot.reply_to(message, "⚠️ AI က စာသားကို နားလည်ဖို့ အခက်အခဲဖြစ်နေပါတယ်။ ဥပမာ - 'ဒီနေ့ ကော်ဖီ ၅၀၀၀ ဖိုးသောက်တယ်' လို့ ရေးပေးပါ။")

print("Expense Tracker Bot is running...")
bot.infinity_polling()