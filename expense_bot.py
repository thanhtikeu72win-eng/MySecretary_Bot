import os
import telebot
import requests
import json
import google.generativeai as genai
from datetime import datetime
from flask import Flask
import threading

# 2. Load Env Vars & n8n Config
TELEGRAM_BOT_TOKEN = os.getenv("EXPENSE_TRACKER_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 

N8N_WEBHOOK_URL = "https://thanhtike72win-n8n-server.hf.space/webhook/add-expense"

# Initialize Bot and Gemini
bot = telebot.TeleBot(EXPENSE_TRACKER_TOKEN)
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# ==========================================
# Render အတွက် Dummy Web Server (ဒီအပိုင်း အသစ်တိုးလာပါတယ်)
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Telegram Bot is running perfectly on Render!"

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
# ==========================================

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_text = message.text
    today_date = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
    You are an expense tracker assistant. Extract the Date, Description, and Amount from the following Burmese text.
    Today's date is: {today_date}
    User Text: "{user_text}"

    Rules:
    - Date must be in YYYY-MM-DD format. If the user doesn't specify a date or says "ဒီနေ့" (today), use {today_date}. 
    - Description should be a short, clear summary. Translate to English or keep it short Burmese.
    - Amount must be a number only (e.g., 5000). Remove currency text.

    Output ONLY a raw JSON object, nothing else. Do not use Markdown blocks.
    Example:
    {{"Date": "2024-03-26", "Description": "Coffee", "Amount": "5000"}}
    """

    try:
        response = model.generate_content(prompt)
        json_string = response.text.strip().replace("```json", "").replace("```", "")
        extracted_data = json.loads(json_string)

        headers = {'Content-Type': 'application/json'}
        n8n_response = requests.post(N8N_WEBHOOK_URL, json=extracted_data, headers=headers)

        if n8n_response.status_code == 200:
            bot_reply = f"✅ စာရင်းသွင်းပြီးပါပြီ!\n\n📅 Date: {extracted_data['Date']}\n📝 Item: {extracted_data['Description']}\n💰 Amount: {extracted_data['Amount']} ks"
            bot.reply_to(message, bot_reply)
        else:
            bot.reply_to(message, f"❌ Google Sheets ဆီသို့ ပို့ရာတွင် အဆင်မပြေဖြစ်နေပါသည်။ (Error: {n8n_response.status_code})")

    except Exception as e:
        print("Error:", e)
        bot.reply_to(message, "⚠️ AI က စာသားကို နားလည်ဖို့ အခက်အခဲဖြစ်နေပါတယ်။ ဥပမာ - 'ဒီနေ့ ကော်ဖီ ၅၀၀၀ ဖိုးသောက်တယ်' လို့ ရေးပေးပါ။")

if __name__ == "__main__":
    # Web Server ကို နောက်ကွယ်ကနေ အရင် Run ပါမယ်
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
    
    # ပြီးမှ Telegram Bot ကို Run ပါမယ်
    print("Expense Tracker Bot is running...")
    bot.infinity_polling()