import os
import logging
import tempfile
import requests
import json
from datetime import datetime, timedelta, timezone
from flask import Flask
from threading import Thread
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, BotCommand
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, CallbackQueryHandler, filters

# Gemini & Pinecone Imports
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# 🆕 Google Calendar Imports
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# 1. Setup Logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Load Env Vars
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mysecretary79-bot")

# 🆕 Google Calendar Env Vars
GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

MMT = timezone(timedelta(hours=6, minutes=30))  # Myanmar Timezone

# Debug Check
print(f"DEBUG CHECK: TELEGRAM_BOT_TOKEN is {'✅ OK' if TELEGRAM_BOT_TOKEN else '❌ MISSING'}")
print(f"DEBUG CHECK: GOOGLE_API_KEY is {'✅ OK' if GOOGLE_API_KEY else '❌ MISSING'}")
print(f"DEBUG CHECK: PINECONE_INDEX_NAME is {'✅ OK' if PINECONE_INDEX_NAME else '❌ MISSING'}")
print(f"DEBUG CHECK: PINECONE_API_KEY is {'✅ OK' if PINECONE_API_KEY else '❌ MISSING'}")
print(f"DEBUG CHECK: GOOGLE_CALENDAR_ID is {'✅ OK' if GOOGLE_CALENDAR_ID else '❌ MISSING'}")
print(f"DEBUG CHECK: GOOGLE_SERVICE_ACCOUNT_JSON is {'✅ OK' if GOOGLE_SERVICE_ACCOUNT_JSON else '❌ MISSING'}")

# 🔒 SECURITY LOCK
ALLOWED_USER_ID = os.getenv("ALLOWED_USER_ID")

# 3. Global Vars
vector_store = None
llm = None
pinecone_index = None
calendar_service = None  # 🆕 Google Calendar Service

def init_services():
    global vector_store, llm, pinecone_index, calendar_service
    try:
        # Gemini LLM
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
            logger.info("✅ Gemini LLM Initialized")

        # Pinecone
        if PINECONE_API_KEY and GOOGLE_API_KEY:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            pinecone_index = pc.Index(PINECONE_INDEX_NAME)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)
            vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings)
            logger.info("✅ Pinecone Services Initialized")

        # 🆕 Google Calendar
        if GOOGLE_SERVICE_ACCOUNT_JSON and GOOGLE_CALENDAR_ID:
            creds_info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
            credentials = service_account.Credentials.from_service_account_info(
                creds_info,
                scopes=['https://www.googleapis.com/auth/calendar']
            )
            calendar_service = build('calendar', 'v3', credentials=credentials, cache_discovery=False)
            logger.info("✅ Google Calendar Initialized")

    except Exception as e:
        logger.error(f"❌ Service Init Error: {e}")

# ---------------------------------------------------------
# 🆕 GOOGLE CALENDAR FUNCTIONS
# ---------------------------------------------------------

def create_calendar_event(event_name, start_time, end_time, description=""):
    """Create a new event in Google Calendar"""
    try:
        if not calendar_service:
            return None, "Calendar service not initialized"
        
        event = {
            'summary': event_name,
            'description': description,
            'start': {
                'dateTime': start_time,
                'timeZone': 'Asia/Yangon'
            },
            'end': {
                'dateTime': end_time,
                'timeZone': 'Asia/Yangon'
            },
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'popup', 'minutes': 60},  # 1 hour before
                    {'method': 'popup', 'minutes': 15},  # 15 min before
                ],
            },
        }
        
        result = calendar_service.events().insert(
            calendarId=GOOGLE_CALENDAR_ID,
            body=event
        ).execute()
        
        return result, None
    except HttpError as e:
        logger.error(f"Calendar HTTP Error: {e}")
        return None, f"HTTP Error: {e}"
    except Exception as e:
        logger.error(f"Calendar Create Error: {e}")
        return None, str(e)

def list_upcoming_events(max_results=10):
    """List upcoming events from Google Calendar"""
    try:
        if not calendar_service:
            return None
        
        now = datetime.now(MMT).isoformat()
        events_result = calendar_service.events().list(
            calendarId=GOOGLE_CALENDAR_ID,
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        return events
    except Exception as e:
        logger.error(f"Calendar List Error: {e}")
        return None

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def get_weather_card(city_name):
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url, timeout=10).json()
        if not geo_res.get('results'): return None
        
        lat = geo_res['results'][0]['latitude']
        lon = geo_res['results'][0]['longitude']
        name = geo_res['results'][0]['name']
        country = geo_res['results'][0]['country']

        w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m&timezone=auto"
        w_res = requests.get(w_url, timeout=10).json()
        curr = w_res['current']

        aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=us_aqi,pm2_5"
        aqi_res = requests.get(aqi_url, timeout=10).json()
        curr_aqi = aqi_res.get('current', {'us_aqi': 'N/A', 'pm2_5': 'N/A'})
        
        code = curr['weather_code']
        if code <= 3: status = "Sunny/Cloudy 🌤️"
        elif code <= 67: status = "Rainy 🌧️"
        elif code <= 99: status = "Stormy ⛈️"
        else: status = "Normal"

        return {
            "name": name, "country": country,
            "temp": curr['temperature_2m'],
            "feels": curr['apparent_temperature'],
            "wind": curr['wind_speed_10m'],
            "rain": curr['precipitation'],
            "status": status,
            "us_aqi": curr_aqi['us_aqi'],
            "pm25": curr_aqi['pm2_5']
        }
    except Exception as e:
        logger.error(f"Weather Error: {e}")
        return None

def get_cbm_card_data():
    try:
        cbm = requests.get("https://forex.cbm.gov.mm/api/latest", timeout=10).json()
        return {
            "date": cbm['info'],
            "rates": cbm['rates']
        }
    except Exception as e:
        logger.error(f"Currency Error: {e}")
        return None

# ---------------------------------------------------------
# Keyboards
# ---------------------------------------------------------

MAIN_MENU = ReplyKeyboardMarkup(
    [[KeyboardButton("🧠 My Brain"), KeyboardButton("🤖 AI Assistant")],
     [KeyboardButton("📅 My Schedule"), KeyboardButton("⚡ Utilities")]], 
    resize_keyboard=True, is_persistent=True
)

AI_TOOLS_MENU = ReplyKeyboardMarkup(
    [[KeyboardButton("✉️ Email Draft"), KeyboardButton("📝 Summarize")],
     [KeyboardButton("🇬🇧⇄🇲🇲 Translate"), KeyboardButton("🧾 Report")],
     [KeyboardButton("🔙 Main Menu")]], 
    resize_keyboard=True, is_persistent=True
)

SCHEDULE_MENU = ReplyKeyboardMarkup(
    [[KeyboardButton("➕ Reminder သစ်"), KeyboardButton("📋 စာရင်းကြည့်")],
     [KeyboardButton("✅ Task Done"), KeyboardButton("🔙 Main Menu")]], 
    resize_keyboard=True, is_persistent=True
)

UTILS_MENU = ReplyKeyboardMarkup(
    [[KeyboardButton("🌦️ Weather"), KeyboardButton("💰 Currency")],
     [KeyboardButton("⚙️ Settings"), KeyboardButton("ℹ️ About Secretary")],
     [KeyboardButton("🔙 Main Menu")]], 
    resize_keyboard=True, is_persistent=True
)

SETTINGS_MENU = ReplyKeyboardMarkup(
    [[KeyboardButton("🔄 Change Persona"), KeyboardButton("🗑️ Clear Memory")],
     [KeyboardButton("🔙 Back")]], 
    resize_keyboard=True, is_persistent=True
)

BACK_BTN = ReplyKeyboardMarkup([[KeyboardButton("🔙 Back")]], resize_keyboard=True, is_persistent=True)

# ---------------------------------------------------------
# Handlers
# ---------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['section'] = 'main'
    context.user_data['mode'] = None
    if 'persona' not in context.user_data: context.user_data['persona'] = 'cute'
    
    commands = [
        BotCommand("start", "🏠 Main Menu"),
        BotCommand("weather", "🌦️ Check Weather"),
        BotCommand("currency", "💰 Check Rates"),
    ]
    await context.bot.set_my_commands(commands)
    
    await update.message.reply_text("မင်္ဂလာပါ ဆရာ့ အတွင်းရေးမှူးမလေး အဆင်သင့်ရှိနေပါတယ်ရှင်။ 👩‍💼\n\nဒီနေ့ ဘာကူညီပေးရမလဲ?", reply_markup=MAIN_MENU)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = update.message.text
        user_mode = context.user_data.get('mode')
        section = context.user_data.get('section', 'main')
        
        # Navigation
        if text == "🔙 Back" or text == "🔙 Main Menu" or text == "/start":
            context.user_data['mode'] = None
            if section == 'settings':
                context.user_data['section'] = 'utils'
                await update.message.reply_text("Utilities Menu", reply_markup=UTILS_MENU)
            else:
                context.user_data['section'] = 'main'
                await update.message.reply_text("Main Menu", reply_markup=MAIN_MENU)
            return

        # Commands
        if text == "/weather":
            context.user_data['section'] = 'utils'
            context.user_data['mode'] = 'check_weather'
            await update.message.reply_text("🌦️ ဘယ်မြို့ရဲ့ ရာသီဥတုကို ကြည့်ပေးရမလဲ ဆရာ? (Naypyitaw,Yangon, Mandalay)", reply_markup=BACK_BTN)
            return
        
        if text == "/currency":
            text = "💰 Currency" 

        # ==========================================
        # ACTION MODES LOGIC
        # ==========================================
        if user_mode == 'check_weather':
            city = text
            await update.message.reply_text(f"🔍 {city} အတွက် Dashboard လေး ထုတ်ပေးနေပါတယ်ရှင်...", reply_markup=UTILS_MENU)
            
            w_data = get_weather_card(city)
            if w_data:
                msg = f"🌤️ <b>WEATHER DASHBOARD</b>\n"
                msg += f"📍 <b>{w_data['name']}, {w_data['country']}</b>\n"
                msg += "━━━━━━━━━━━━━━━━━━\n"
                msg += f"🌡️ Temp  : <b>{w_data['temp']}°C</b> (Feels {w_data['feels']}°C)\n"
                msg += f"🏭 AQI   : <b>{w_data['us_aqi']} US AQI</b>\n"
                msg += f"😷 PM2.5 : <b>{w_data['pm25']} μg/m³</b>\n"
                msg += f"💨 Wind  : <b>{w_data['wind']} km/h</b>\n"
                msg += f"💧 Rain  : <b>{w_data['rain']} mm</b>\n"
                msg += "━━━━━━━━━━━━━━━━━━\n"
                msg += f"💡 Status: {w_data['status']}"
                await update.message.reply_text(msg, parse_mode="HTML", reply_markup=UTILS_MENU)
            else:
                await update.message.reply_text("❌ မြို့နာမည် ရှာမတွေ့ပါရှင်။ English လို သေချာရိုက်ပေးပါနော် Boss။", reply_markup=UTILS_MENU)
            context.user_data['mode'] = None
            return

        # ==========================================
        # 🆕 NEW: Google Calendar Direct Integration
        # ==========================================
        elif user_mode == 'add_calendar_event':
            await update.message.reply_text("⏳ Google Calendar ထဲကို ချိတ်ဆက်ထည့်သွင်းပေးနေပါတယ်ရှင်...", reply_markup=SCHEDULE_MENU)
            try:
                if not calendar_service:
                    await update.message.reply_text("❌ Calendar Service ချိတ်ဆက်မထားပါရှင်။ Environment Variables စစ်ပေးပါ။")
                    context.user_data['mode'] = None
                    return
                
                current_time = datetime.now(MMT).strftime('%Y-%m-%d %H:%M:%S')
                day_of_week = datetime.now(MMT).strftime('%A')
                
                # Ask Gemini to format the natural language into JSON
                prompt = f"""
You are an AI assistant that extracts calendar event details from Burmese/English text. Return ONLY a valid JSON object (NO markdown, NO code blocks, NO explanation).

Current DateTime in Myanmar (MMT +06:30): {current_time}
Current Day: {day_of_week}

Rules:
1. eventName: Brief title of the event (can be in Burmese or English)
2. startTime: Event start time in format "YYYY-MM-DDTHH:MM:SS+06:30"
3. endTime: Event end time (default to 1 hour after start if not specified)
4. description: Brief description or empty string

Examples:
- "မနက်ဖြန် နေ့လည် ၂ နာရီ Meeting" → start: tomorrow 14:00:00, end: tomorrow 15:00:00
- "Tonight 8pm dinner" → start: today 20:00:00, end: today 21:00:00
- "Next Monday 10am doctor appointment" → start: next monday 10:00:00

User Text: "{text}"

Return JSON only:
"""
                
                ai_response = llm.invoke(prompt)
                ai_text = ai_response.content.strip()
                
                # Clean up markdown if Gemini adds it
                if ai_text.startswith("```json"): ai_text = ai_text[7:]
                if ai_text.startswith("```"): ai_text = ai_text[3:]
                if ai_text.endswith("```"): ai_text = ai_text[:-3]
                ai_text = ai_text.strip()
                
                logger.info(f"AI Parsed: {ai_text}")
                
                # Parse JSON
                event_data = json.loads(ai_text)
                
                # Create event in Google Calendar
                result, error = create_calendar_event(
                    event_name=event_data.get('eventName', 'Untitled Event'),
                    start_time=event_data.get('startTime'),
                    end_time=event_data.get('endTime'),
                    description=event_data.get('description', '')
                )
                
                if result:
                    # Format display time
                    start_dt = datetime.fromisoformat(event_data['startTime'].replace('+06:30', ''))
                    display_time = start_dt.strftime('%Y-%m-%d %I:%M %p')
                    
                    success_msg = f"✅ <b>Google Calendar ထဲ မှတ်သားပြီးပါပြီ Boss!</b>\n\n"
                    success_msg += f"📌 <b>ပွဲအမည်:</b> {event_data.get('eventName')}\n"
                    success_msg += f"🕐 <b>အချိန်:</b> {display_time}\n"
                    
                    if event_data.get('description'):
                        success_msg += f"📝 <b>မှတ်ချက်:</b> {event_data.get('description')}\n"
                    
                    success_msg += f"\n🔔 မိနစ် ၆၀ နှင့် ၁၅ မိနစ်အလို သတိပေးပါမယ်ရှင်။\n"
                    success_msg += f"\n🔗 <a href=\"{result.get('htmlLink')}\">Calendar မှာ ကြည့်ရန်</a>"
                    
                    await update.message.reply_text(success_msg, parse_mode="HTML", disable_web_page_preview=True)
                else:
                    await update.message.reply_text(f"❌ Calendar ထဲ ထည့်ရာမှာ အမှားဖြစ်သွားပါတယ်ရှင်။\n\nError: {error}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON Parse Error: {e}")
                await update.message.reply_text("❌ AI က အချိန်ကို မှန်မှန်ကန်ကန် ခွဲမထုတ်နိုင်ပါဘူးရှင်။\n\nဥပမာ - \"မနက်ဖြန် မနက် ၁၀ နာရီ Meeting\" လို ပိုရှင်းအောင် ပြန်ရေးပေးပါနော်။")
            except Exception as e:
                logger.error(f"Calendar Event Error: {e}")
                await update.message.reply_text(f"❌ Error ဖြစ်သွားပါတယ်ရှင်။\n\nDetails: {str(e)[:200]}")
            
            context.user_data['mode'] = None
            return

        elif user_mode == 'add_task':
            tasks = context.user_data.get('tasks', [])
            tasks.append(text)
            context.user_data['tasks'] = tasks
            await update.message.reply_text("✅ မှတ်သားလိုက်ပါပြီ Boss။", reply_markup=SCHEDULE_MENU)
            context.user_data['mode'] = None
            return

        elif user_mode == 'remove_task':
            tasks = context.user_data.get('tasks', [])
            if text.isdigit() and 1 <= int(text) <= len(tasks):
                removed = tasks.pop(int(text)-1)
                context.user_data['tasks'] = tasks
                await update.message.reply_text(f"✅ စာရင်းမှ ပယ်ဖျက်လိုက်ပါပြီရှင်။", reply_markup=SCHEDULE_MENU)
            else:
                await update.message.reply_text("❌ နံပါတ် မှားနေပါတယ်ရှင်။", reply_markup=SCHEDULE_MENU)
            context.user_data['mode'] = None
            return

        elif user_mode in ['email', 'summarize', 'translate', 'report']:
            await call_ai_direct(update, context, f"Task: {user_mode}. Content: {text}")
            context.user_data['mode'] = None
            return

        elif user_mode == 'add_link':
            await process_link(update, context, text)
            context.user_data['mode'] = None
            return

        # ==========================================
        # MAIN MENU & SUB MENU BUTTON LOGIC
        # ==========================================
        if text == "🧠 My Brain":
            context.user_data['section'] = 'brain'
            keyboard = [[InlineKeyboardButton("📥 Add PDF/Word", callback_data="add_doc"), InlineKeyboardButton("🔗 Add Link", callback_data="add_link")], [InlineKeyboardButton("📊 Stats", callback_data="list_mem"), InlineKeyboardButton("🗑️ Delete Data", callback_data="del_data")]]
            await update.message.reply_text("🧠 **My Brain Panel**", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
            return

        elif text == "🤖 AI Assistant":
            context.user_data['section'] = 'ai_assistant'
            await update.message.reply_text("🤖 **မင်္ဂလာပါ၊ ကျွန်မက ဆရာရဲ့ AI Assistant ပါရှင် မေးခွန်းမေးမြန်းနိုင်ပါတယ်ရှင့်**", reply_markup=AI_TOOLS_MENU)
            return

        elif text == "📅 My Schedule":
            context.user_data['section'] = 'schedule'
            await update.message.reply_text("📅 **My Schedule Panel**\n\nGoogle Calendar နဲ့ ချိတ်ဆက်ထားပါတယ်ရှင်။", reply_markup=SCHEDULE_MENU)
            return

        elif text == "⚡ Utilities":
            context.user_data['section'] = 'utils'
            await update.message.reply_text("⚡ **Utilities**", reply_markup=UTILS_MENU)
            return

        # 🆕 Schedule Menu Handlers
        elif text == "➕ Reminder သစ်":
            context.user_data['mode'] = 'add_calendar_event'
            await update.message.reply_text(
                "📅 ဘာအစီအစဉ် ရှိလဲ Boss? အချိန်နဲ့တကွ ပြောပြပေးပါ။\n\n"
                "<b>ဥပမာ:</b>\n"
                "• မနက်ဖြန် နေ့လည် ၂ နာရီ Meeting\n"
                "• Tonight 8pm dinner with family\n"
                "• Next Monday 10am doctor appointment\n"
                "• ၂၅ ရက် နံနက် ၉ နာရီ စာချုပ်လက်မှတ်ထိုးမယ်",
                parse_mode="HTML",
                reply_markup=BACK_BTN
            )
            return

        elif text == "📋 စာရင်းကြည့်":
            await update.message.reply_text("🔍 Google Calendar မှ Events များ ဆွဲထုတ်နေပါတယ်ရှင်...", reply_markup=SCHEDULE_MENU)
            
            events = list_upcoming_events(max_results=10)
            if events is None:
                await update.message.reply_text("❌ Calendar ချိတ်ဆက်မှု အမှားဖြစ်နေပါတယ်ရှင်။")
            elif not events:
                await update.message.reply_text("📭 လာမည့်ရက်များတွင် အစီအစဉ် မရှိသေးပါရှင်။")
            else:
                msg = "📋 <b>လာမည့် အစီအစဉ်များ</b>\n"
                msg += "━━━━━━━━━━━━━━━━━━\n\n"
                for i, event in enumerate(events, 1):
                    start = event['start'].get('dateTime', event['start'].get('date'))
                    try:
                        start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                        display_time = start_dt.strftime('%m/%d %I:%M %p')
                    except:
                        display_time = start
                    
                    msg += f"<b>{i}. {event.get('summary', 'Untitled')}</b>\n"
                    msg += f"   🕐 {display_time}\n\n"
                
                await update.message.reply_text(msg, parse_mode="HTML", reply_markup=SCHEDULE_MENU)
            return

        elif text == "✅ Task Done":
            await update.message.reply_text(
                "✅ Great job Boss! 👏\n\nGoogle Calendar ထဲက event တစ်ခု ဖျက်ချင်ရင် Calendar app ထဲမှာ တိုက်ရိုက် ဖျက်ပေးပါနော်ရှင်။",
                reply_markup=SCHEDULE_MENU
            )
            return

        if section == 'utils' or text == "💰 Currency" or text == "🌦️ Weather":
            if text == "🌦️ Weather":
                context.user_data['mode'] = 'check_weather'
                await update.message.reply_text("🌦️ ဘယ်မြို့ရဲ့ရာသီဥတုကို ကြည့်ပေးရမလဲ ဆရာ? (Naypyitaw,Yangon, Mandalay)", reply_markup=BACK_BTN)
                return
            
            elif text == "💰 Currency":
                await update.message.reply_text("💰 **ဗဟိုဘဏ်ပေါက်ဈေး (CBM Rate) ကို ထုတ်ပေးနေပါတယ်ရှင်...**", reply_markup=UTILS_MENU)
                cbm_data = get_cbm_card_data()
                if cbm_data:
                    msg = f"<b>🏦 CBM EXCHANGE RATES</b>\n"
                    msg += f"📅 <i>{cbm_data['date']}</i>\n\n"
                    msg += "<b>💵 ငွေလဲနှုန်း (Official)</b>\n"
                    msg += "<pre>"
                    msg += "  CURRENCY  |    RATE    \n"
                    msg += "------------+------------\n"
                    msg += f"  🇺🇸 USD    |  {cbm_data['rates']['USD']:<8}\n"
                    msg += f"  🇪🇺 EUR    |  {cbm_data['rates']['EUR']:<8}\n"
                    msg += f"  🇸🇬 SGD    |  {cbm_data['rates']['SGD']:<8}\n"
                    msg += f"  🇹🇭 THB    |  {cbm_data['rates']['THB']:<8}\n"
                    msg += "</pre>\n"
                    msg += f"💡 <i>Source: Central Bank of Myanmar</i>"
                    await update.message.reply_text(msg, parse_mode="HTML", reply_markup=UTILS_MENU)
                else:
                    await update.message.reply_text("❌ CBM Data Error", reply_markup=UTILS_MENU)
                return

            elif text == "⚙️ Settings":
                context.user_data['section'] = 'settings'
                await update.message.reply_text("⚙️ **Settings**", reply_markup=SETTINGS_MENU)
                return
            
            elif text == "ℹ️ About Secretary":
                about_msg = """
ℹ️ **About Your Secretary Bot** 👩‍💼

ကျွန်မက ဆရာရဲ့ ကိုယ်ပိုင် Digital အတွင်းရေးမှူးမလေး ဖြစ်ပါတယ်ရှင်။
ကျွန်မ လုပ်ပေးနိုင်တာတွေကတော့ -

1.  **🧠 My Brain:** စာရွက်စာတမ်း (PDF/Word) တွေကို ဖတ်ပြီး မှတ်ထားပေးပါတယ်။
2.  **📅 My Schedule:** Google Calendar နဲ့ ချိတ်ဆက်ပြီး Reminder တွေ မှတ်ပေးပါတယ်။
3.  **🌦️ Weather:** မိုးလေဝသ အခြေအနေ ကြည့်ပေးပါတယ်။
4.  **💰 Currency:** ဗဟိုဘဏ် ပေါက်ဈေးတွေကို ကြည့်ပေးပါတယ်။
5.  **🤖 AI Tools:** Email ရေးခြင်း၊ ဘာသာပြန်ခြင်း ကူညီပေးပါတယ်။
                """
                await update.message.reply_text(about_msg.strip(), reply_markup=UTILS_MENU)
                return

        # AI Chat Fallback
        if section == 'ai_assistant' and not user_mode:
            if not vector_store:
                await update.message.reply_text("Database Error ပါရှင်။")
                return
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            try:
                docs = vector_store.similarity_search(text, k=3)
                context_str = "\n".join([d.page_content for d in docs])
                prompt = f"Role: You are a polite female secretary. Context: {context_str}\n\nQ: {text}\n\nAns (Burmese):"
                response = llm.invoke(prompt)
                await update.message.reply_text(response.content)
            except Exception as e:
                logger.error(f"AI Error: {e}")
                await update.message.reply_text("Error")
            return
            
        await update.message.reply_text("Menu က ခလုတ်လေးတွေ ရွေးပေးပါနော် Boss။", reply_markup=MAIN_MENU)

    except Exception as e:
        logger.error(f"Global Handler Error: {e}")
        context.user_data['section'] = 'main'
        context.user_data['mode'] = None
        await update.message.reply_text("⚠️ Error လေးတစ်ခုဖြစ်သွားလို့ Main Menu ကို ပြန်သွားပေးပါမယ်ရှင်။", reply_markup=MAIN_MENU)

async def call_ai_direct(update, context, prompt):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        response = llm.invoke(prompt)
        await update.message.reply_text(response.content)
    except Exception:
        pass

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "add_doc":
        await query.edit_message_text("📥 PDF/Word ဖိုင်လေး ပို့ပေးပါရှင်။")
    elif query.data == "add_link":
        context.user_data['mode'] = 'add_link'
        await query.edit_message_text("🔗 Link လေး ပို့ပေးပါနော်။")
    elif query.data == "del_data":
        context.user_data['mode'] = 'delete_data'
        await query.edit_message_text("🗑️ ဖျက်ချင်တဲ့ ဖိုင်နာမည် ပို့ပေးပါရှင်။")
    elif query.data == "list_mem": 
        stats = pinecone_index.describe_index_stats()
        await query.edit_message_text(f"📊 Vectors: {stats.get('total_vector_count')}")

async def process_link(update, context, url):
    msg = await update.message.reply_text("🔗 Processing...")
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(docs)
        for t in texts:
            t.metadata = {"source": url}
        vector_store.add_documents(texts)
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text="✅ Done.")
    except Exception:
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text="Error")

async def handle_document(update, context):
    msg = await update.message.reply_text("📥 Processing...")
    try:
        file = await context.bot.get_file(update.message.document.file_id)
        fname = update.message.document.file_name
        with tempfile.NamedTemporaryFile(delete=True, suffix=os.path.splitext(fname)[1]) as tmp:
            await file.download_to_drive(custom_path=tmp.name)
            if fname.endswith(".pdf"):
                loader = PyPDFLoader(tmp.name)
            else:
                loader = Docx2txtLoader(tmp.name)
            texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
            for t in texts:
                t.metadata = {"source": fname}
            vector_store.add_documents(texts)
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text=f"✅ Saved.")
    except Exception:
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text="Error")

# Flask & Main
flask_app = Flask('')
@flask_app.route('/') 
def home(): return "Bot Online"
def run_flask(): flask_app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))

if __name__ == '__main__':
    Thread(target=run_flask).start()
    init_services()
    if TELEGRAM_BOT_TOKEN:
        app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        app.add_handler(CommandHandler('start', start))
        app.add_handler(CommandHandler('weather', lambda u,c: handle_message(u,c)))
        app.add_handler(CommandHandler('currency', lambda u,c: handle_message(u,c)))
        app.add_handler(CallbackQueryHandler(handle_callback_query))
        app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
        app.run_polling()
