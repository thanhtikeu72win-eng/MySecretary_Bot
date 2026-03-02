import os
import logging
import tempfile
import requests
import json
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

# 1. Setup Logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Load Env Vars
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mysecretary79-bot")

# Debug Check
print(f"DEBUG CHECK: TELEGRAM_BOT_TOKEN is {'✅ OK' if TELEGRAM_BOT_TOKEN else '❌ MISSING'}")
print(f"DEBUG CHECK: GOOGLE_API_KEY is {'✅ OK' if GOOGLE_API_KEY else '❌ MISSING'}")
print(f"DEBUG CHECK: PINECONE_INDEX_NAME is {'✅ OK' if PINECONE_INDEX_NAME else '❌ MISSING'}")
print(f"DEBUG CHECK: PINECONE_API_KEY is {'✅ OK' if PINECONE_API_KEY else '❌ MISSING'}")

# 🔒 SECURITY LOCK
# Get User ID from Env, if empty allow everyone (Not Recommended for Private Data)
ALLOWED_USER_ID = os.getenv("ALLOWED_USER_ID")

# 3. Global Vars
vector_store = None
llm = None
pinecone_index = None

def init_services():
    global vector_store, llm, pinecone_index
    try:
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

        if PINECONE_API_KEY and GOOGLE_API_KEY:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            pinecone_index = pc.Index(PINECONE_INDEX_NAME)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)
            vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings)
            logger.info("✅ Pinecone Services Initialized")
    except Exception as e:
        logger.error(f"❌ Service Init Error: {e}")

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

        # Action Modes
        if user_mode == 'check_weather':
            city = text
            await update.message.reply_text(f"🔍 {city} အတွက် Dashboard လေး ထုတ်ပေးနေပါတယ်ရှင်...", reply_markup=UTILS_MENU)
            
            w_data = get_weather_card(city)
            if w_data:
                msg = f"🌤️ <b>WEATHER DASHBOARD</b>\n"
                msg += f"📍 <b>{w_data['name']}, {w_data['country']}</b>\n"
                msg += "━━━━━━━━━━━━━━━━━━\n"
                msg += f"🌡️ Temp  : <b>{w_data['temp']}°C</b> (Feels {w_data['feels']}°C)\n"
                # Fixed: Removed '+' sign to be accurate to API data
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

        # Main Menu Logic
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
            tasks = context.user_data.get('tasks', [])
            task_str = "\n".join([f"{i+1}. {t}" for i, t in enumerate(tasks)]) if tasks else "ဒီနေ့အတွက် Reminder ဘာမှမရှိသေးပါဘူးရှင်။"
            await update.message.reply_text(f"📅 **Today's Plan:**\n\n{task_str}", reply_markup=SCHEDULE_MENU)
            return

        elif text == "⚡ Utilities":
            context.user_data['section'] = 'utils'
            await update.message.reply_text("⚡ **Utilities**", reply_markup=UTILS_MENU)
            return

        # Sub Menu Logic
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

1.  **🧠 My Brain:** စာရွက်စာတမ်း (PDF/Word) တွေကို ဖတ်ပြီး မှတ်ထားပေးပါတယ်။ မေးသမျှကို ပြန်ဖြေပေးပါတယ်။
2.  **📅 My Schedule:** နေ့စဉ် လုပ်စရာတွေကို မှတ်ပေး၊ သတိပေးပါတယ်။
3.  **🌦️ Weather:** မိုးလေဝသ အခြေအနေကို အချိန်နဲ့တပြေးညီ ကြည့်ပေးပါတယ်။
4.  **💰 Currency:** ဗဟိုဘဏ် ပေါက်ဈေးတွေကို ကြည့်ပေးပါတယ်။
5.  **🤖 AI Tools:** Email ရေးခြင်း၊ ဘာသာပြန်ခြင်း၊ Report ရေးခြင်းတို့ကို ကူညီပေးပါတယ်။

ဆရာ စိတ်တိုင်းကျ ခိုင်းစေနိုင်ပါတယ်ရှင်! 💖
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