import os
import logging
import tempfile
import requests
import json
from flask import Flask
from threading import Thread
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton
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

# 3. Global Vars
vector_store = None
llm = None
pinecone_index = None
MARKET_RATE_USD = 5200  # Default Estimate

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

def get_aqi_status(aqi):
    if aqi <= 50: return "🟢 Good (သန့်ရှင်း)"
    elif aqi <= 100: return "🟡 Moderate (အသင့်အတင့်)"
    elif aqi <= 150: return "🟠 Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "🔴 Unhealthy (ကျန်းမာရေး ထိခိုက်နိုင်)"
    else: return "🟣 Very Unhealthy (အန္တရာယ်ရှိ)"

def get_weather_data(city_name):
    try:
        # Geocoding
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url).json()
        if not geo_res.get('results'): return None
        
        lat = geo_res['results'][0]['latitude']
        lon = geo_res['results'][0]['longitude']
        name = geo_res['results'][0]['name']
        country = geo_res['results'][0]['country']

        # Weather & Air Quality
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,weather_code,wind_speed_10m,wind_direction_10m&hourly=uv_index&timezone=auto"
        aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=us_aqi,pm2_5&timezone=auto"
        
        w_res = requests.get(url).json()
        a_res = requests.get(aqi_url).json()
        
        curr = w_res['current']
        curr_aqi = a_res['current']
        
        return {
            "name": name, "country": country,
            "temp": curr['temperature_2m'],
            "feels_like": curr['apparent_temperature'],
            "humidity": curr['relative_humidity_2m'],
            "wind_speed": curr['wind_speed_10m'],
            "wind_dir": curr['wind_direction_10m'],
            "aqi": curr_aqi['us_aqi'],
            "pm25": curr_aqi['pm2_5']
        }
    except Exception as e:
        logger.error(f"Weather Error: {e}")
        return None

def get_currency_data():
    try:
        cbm = requests.get("https://forex.cbm.gov.mm/api/latest").json()
        cbm_rates = cbm['rates']
        usd_official = float(cbm_rates['USD'].replace(',', ''))
        
        # Calculate Market Estimates based on the user-set USD rate
        factor = MARKET_RATE_USD / usd_official
        
        market_usd = MARKET_RATE_USD
        market_eur = float(cbm_rates['EUR'].replace(',', '')) * factor
        market_sgd = float(cbm_rates['SGD'].replace(',', '')) * factor
        market_thb = float(cbm_rates['THB'].replace(',', '')) * factor
        
        return {
            "date": cbm['info'],
            "official": {"USD": usd_official, "EUR": cbm_rates['EUR'], "SGD": cbm_rates['SGD'], "THB": cbm_rates['THB']},
            "market": {"USD": market_usd, "EUR": market_eur, "SGD": market_sgd, "THB": market_thb}
        }
    except Exception as e:
        logger.error(f"Currency Error: {e}")
        return None

# ---------------------------------------------------------
# Keyboards
# ---------------------------------------------------------

MAIN_MENU = ReplyKeyboardMarkup(
    [[KeyboardButton("🧠 My Brain"), KeyboardButton("🤖 AI Assistant")],
     [KeyboardButton("📅 My Schedule"), KeyboardButton("⚡ Utilities")]], resize_keyboard=True
)

AI_TOOLS_MENU = ReplyKeyboardMarkup(
    [[KeyboardButton("✉️ Email Draft"), KeyboardButton("📝 Summarize")],
     [KeyboardButton("🇬🇧⇄🇲🇲 Translate"), KeyboardButton("🧾 Report")],
     [KeyboardButton("🔙 Main Menu")]], resize_keyboard=True
)

SCHEDULE_MENU = ReplyKeyboardMarkup(
    [[KeyboardButton("➕ Reminder သစ်"), KeyboardButton("📋 စာရင်းကြည့်")],
     [KeyboardButton("✅ Task Done"), KeyboardButton("🔙 Main Menu")]], resize_keyboard=True
)

UTILS_MENU = ReplyKeyboardMarkup(
    [[KeyboardButton("🌦️ Weather"), KeyboardButton("💰 Currency")],
     [KeyboardButton("⚙️ Settings"), KeyboardButton("ℹ️ About Secretary")],
     [KeyboardButton("🔙 Main Menu")]], resize_keyboard=True
)

SETTINGS_MENU = ReplyKeyboardMarkup(
    [[KeyboardButton("🔄 Change Persona"), KeyboardButton("🗑️ Clear Memory")],
     [KeyboardButton("✏️ Set Market Rate"), KeyboardButton("🔙 Back")]], resize_keyboard=True
)

BACK_BTN = ReplyKeyboardMarkup([[KeyboardButton("🔙 Back")]], resize_keyboard=True)

# ---------------------------------------------------------
# Handlers
# ---------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['section'] = 'main'
    context.user_data['mode'] = None
    if 'persona' not in context.user_data: context.user_data['persona'] = 'cute'
    await update.message.reply_text("မင်္ဂလာပါ Boss! ရှင့်ရဲ့ Secretary Bot လေး အဆင်သင့်ပါရှင်။", reply_markup=MAIN_MENU)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = update.message.text
        user_mode = context.user_data.get('mode')
        section = context.user_data.get('section', 'main') # Default to main if None
        persona = context.user_data.get('persona', 'cute')

        # --- 1. Global Back Button Logic (Highest Priority) ---
        if text == "🔙 Back" or text == "🔙 Main Menu":
            # Reset Mode
            context.user_data['mode'] = None
            
            # Logic for where to go back
            if section == 'settings':
                context.user_data['section'] = 'utils'
                await update.message.reply_text("⚡ Utilities Menu", reply_markup=UTILS_MENU)
            elif section == 'utils':
                context.user_data['section'] = 'main'
                await update.message.reply_text("🏠 Main Menu", reply_markup=MAIN_MENU)
            elif section == 'schedule':
                context.user_data['section'] = 'main'
                await update.message.reply_text("🏠 Main Menu", reply_markup=MAIN_MENU)
            elif section == 'ai_assistant':
                context.user_data['section'] = 'main'
                await update.message.reply_text("🏠 Main Menu", reply_markup=MAIN_MENU)
            else:
                context.user_data['section'] = 'main'
                await update.message.reply_text("🏠 Main Menu", reply_markup=MAIN_MENU)
            return

        # --- 2. Action Modes (Waiting for Input) ---
        if user_mode == 'add_link':
            if text.startswith("http"): await process_link(update, context, text)
            else: await update.message.reply_text("❌ Link အမှန်မဟုတ်ပါ", reply_markup=BACK_BTN)
            context.user_data['mode'] = None; return

        elif user_mode == 'set_market_rate':
            global MARKET_RATE_USD
            if text.isdigit():
                MARKET_RATE_USD = int(text)
                await update.message.reply_text(f"✅ Market Rate (USD) ကို {MARKET_RATE_USD} သို့ ပြောင်းလိုက်ပါပြီ။", reply_markup=SETTINGS_MENU)
            else:
                await update.message.reply_text("❌ ဂဏန်းသီးသန့် ရိုက်ထည့်ပေးပါရှင် (ဥပမာ: 4500)။", reply_markup=SETTINGS_MENU)
            context.user_data['mode'] = None; return

        elif user_mode == 'check_weather':
            city = text
            await update.message.reply_text(f"🔍 {city} မြို့အတွက် ရှာဖွေနေပါတယ်...", reply_markup=UTILS_MENU)
            data = get_weather_data(city)
            if data:
                aqi_status = get_aqi_status(data['aqi'])
                report = f"🌤️ **Weather: {data['name']}**\n"
                report += f"🌡️ Temp: {data['temp']}°C (Feels: {data['feels_like']}°C)\n"
                report += f"💨 Wind: {data['wind_speed']} km/h ({data['wind_dir']}°)\n"
                report += f"🏭 **Air Quality:** {data['aqi']} US AQI\n({aqi_status})\n"
                
                # Simple AI Advice
                try:
                    prompt = f"Weather: {data['temp']}C, AQI: {data['aqi']}. Give 1 short health tip in Burmese."
                    advice = llm.invoke(prompt).content
                    report += f"\n💡 **Tip:** {advice}"
                except: pass
                
                await update.message.reply_text(report, parse_mode="Markdown", reply_markup=UTILS_MENU)
            else:
                await update.message.reply_text("❌ မတွေ့ပါရှင်။", reply_markup=UTILS_MENU)
            context.user_data['mode'] = None; return

        elif user_mode == 'add_task':
            tasks = context.user_data.get('tasks', []); tasks.append(text); context.user_data['tasks'] = tasks
            await update.message.reply_text("✅ Saved.", reply_markup=SCHEDULE_MENU); context.user_data['mode'] = None; return

        elif user_mode == 'remove_task':
            tasks = context.user_data.get('tasks', [])
            if text.isdigit() and 1 <= int(text) <= len(tasks):
                removed = tasks.pop(int(text)-1); context.user_data['tasks'] = tasks
                await update.message.reply_text(f"✅ Removed: {removed}", reply_markup=SCHEDULE_MENU)
            else: await update.message.reply_text("❌ Invalid Number", reply_markup=SCHEDULE_MENU)
            context.user_data['mode'] = None; return
        
        elif user_mode == 'delete_data':
            try:
                pinecone_index.delete(filter={"source": {"$eq": text}})
                await update.message.reply_text(f"🗑️ Deleted: {text}", reply_markup=MAIN_MENU)
            except Exception as e: await update.message.reply_text(f"Error: {e}")
            context.user_data['mode'] = None; return

        elif user_mode in ['email', 'summarize', 'translate', 'report']:
            await call_ai_direct(update, context, f"Task: {user_mode}. Text: {text}")
            context.user_data['mode'] = None; return

        # --- 3. Menu Navigation ---
        
        # Main Menu
        if text == "🧠 My Brain":
            context.user_data['section'] = 'brain'
            keyboard = [[InlineKeyboardButton("📥 Add PDF/Word", callback_data="add_doc"), InlineKeyboardButton("🔗 Add Link", callback_data="add_link")], [InlineKeyboardButton("📊 Stats", callback_data="list_mem"), InlineKeyboardButton("🗑️ Delete Data", callback_data="del_data")]]
            await update.message.reply_text("🧠 **Brain Panel**", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown"); return

        elif text == "🤖 AI Assistant":
            context.user_data['section'] = 'ai_assistant'
            await update.message.reply_text("🤖 **AI Tools**", reply_markup=AI_TOOLS_MENU); return

        elif text == "📅 My Schedule":
            context.user_data['section'] = 'schedule'
            tasks = context.user_data.get('tasks', [])
            task_str = "\n".join([f"{i+1}. {t}" for i, t in enumerate(tasks)]) if tasks else "No tasks."
            await update.message.reply_text(f"📅 **Today's Plan:**\n{task_str}", reply_markup=SCHEDULE_MENU); return

        elif text == "⚡ Utilities":
            context.user_data['section'] = 'utils'
            await update.message.reply_text("⚡ **Utilities**", reply_markup=UTILS_MENU); return

        # Sub Menus
        if section == 'schedule':
            if text == "➕ Reminder သစ်": context.user_data['mode'] = 'add_task'; await update.message.reply_text("Task?", reply_markup=BACK_BTN); return
            elif text == "📋 စာရင်းကြည့်": tasks = context.user_data.get('tasks', []); await update.message.reply_text(f"Tasks:\n" + "\n".join([f"{i+1}. {t}" for i,t in enumerate(tasks)]), reply_markup=SCHEDULE_MENU); return
            elif text == "✅ Task Done": context.user_data['mode'] = 'remove_task'; await update.message.reply_text("Number?", reply_markup=BACK_BTN); return

        if section == 'utils':
            if text == "🌦️ Weather":
                context.user_data['mode'] = 'check_weather'
                await update.message.reply_text("🌦️ City Name? (e.g., Yangon)", reply_markup=BACK_BTN); return
            
            elif text == "💰 Currency":
                data = get_currency_data()
                if data:
                    msg = f"📅 **Date:** {data['date']}\n\n"
                    msg += "```\n"
                    msg += f"{'CURRENCY':<5} | {'🏦 OFFICIAL':<10} | {'⚫ MARKET':<10}\n"
                    msg += "-"*33 + "\n"
                    msg += f"🇺🇸 USD  | {data['official']['USD']:<10,.0f} | {data['market']['USD']:<10,.0f}\n"
                    msg += f"🇪🇺 EUR  | {float(str(data['official']['EUR']).replace(',','')):<10,.0f} | {data['market']['EUR']:<10,.0f}\n"
                    msg += f"🇸🇬 SGD  | {float(str(data['official']['SGD']).replace(',','')):<10,.0f} | {data['market']['SGD']:<10,.0f}\n"
                    msg += f"🇹🇭 THB  | {float(str(data['official']['THB']).replace(',','')):<10,.0f} | {data['market']['THB']:<10,.0f}\n"
                    msg += "```\n"
                    msg += f"💡 **Note:** Market Rate estimated at **{MARKET_RATE_USD}**."
                    await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=UTILS_MENU)
                else:
                    await update.message.reply_text("❌ Data Error", reply_markup=UTILS_MENU)
                return

            elif text == "⚙️ Settings":
                context.user_data['section'] = 'settings'
                await update.message.reply_text("⚙️ **Settings**", reply_markup=SETTINGS_MENU); return
            
            elif text == "ℹ️ About Secretary":
                await update.message.reply_text("ℹ️ **About:**\nSmart Secretary Bot v2.0", reply_markup=UTILS_MENU); return

        if section == 'settings':
            if text == "✏️ Set Market Rate":
                context.user_data['mode'] = 'set_market_rate'
                await update.message.reply_text(f"💵 Current Rate: {MARKET_RATE_USD}\nEnter new rate:", reply_markup=BACK_BTN)
                return
            elif text == "🔄 Change Persona":
                new_p = 'strict' if persona == 'cute' else 'cute'
                context.user_data['persona'] = new_p
                await update.message.reply_text(f"Persona: {new_p}", reply_markup=SETTINGS_MENU); return
            elif text == "🗑️ Clear Memory":
                context.user_data['tasks'] = []
                await update.message.reply_text("Cleared.", reply_markup=SETTINGS_MENU); return

        if section == 'ai_assistant':
            if text == "✉️ Email Draft": context.user_data['mode'] = 'email'; await update.message.reply_text("Topic?", reply_markup=BACK_BTN); return
            elif text == "📝 Summarize": context.user_data['mode'] = 'summarize'; await update.message.reply_text("Text?", reply_markup=BACK_BTN); return
            elif text == "🇬🇧⇄🇲🇲 Translate": context.user_data['mode'] = 'translate'; await update.message.reply_text("Text?", reply_markup=BACK_BTN); return
            elif text == "🧾 Report": context.user_data['mode'] = 'report'; await update.message.reply_text("Topic?", reply_markup=BACK_BTN); return

        # --- Default RAG Chat ---
        if section == 'ai_assistant' and not user_mode:
            if not vector_store: await update.message.reply_text("DB Error"); return
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            try:
                docs = vector_store.similarity_search(text, k=3)
                context_str = "\n".join([d.page_content for d in docs])
                prompt = f"Context: {context_str}\n\nQ: {text}\n\nAnswer in Burmese:"
                response = llm.invoke(prompt)
                await update.message.reply_text(response.content)
            except Exception as e: await update.message.reply_text(f"Error: {e}")
            return
            
        # Fallback
        await update.message.reply_text("Please select a menu option.", reply_markup=MAIN_MENU)

    except Exception as e:
        logger.error(f"Global Handler Error: {e}")
        await update.message.reply_text("⚠️ An error occurred. Resetting...", reply_markup=MAIN_MENU)
        context.user_data['section'] = 'main'
        context.user_data['mode'] = None

async def call_ai_direct(update, context, prompt):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        response = llm.invoke(prompt)
        await update.message.reply_text(response.content)
    except: pass

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == "add_doc": await query.edit_message_text("📥 PDF/Word ပို့ပါ Boss။")
    elif query.data == "add_link": 
        context.user_data['mode'] = 'add_link'
        await query.edit_message_text("🔗 Link ပို့ပါ Boss။")
    elif query.data == "del_data":
        context.user_data['mode'] = 'delete_data'
        await query.edit_message_text("🗑️ ဖျက်လိုသော Source Path ပို့ပါ (Logs မှကြည့်ပါ)။")
    elif query.data == "list_mem":
        stats = pinecone_index.describe_index_stats()
        await query.edit_message_text(f"📊 Stats:\nVectors: {stats.get('total_vector_count')}")

async def process_link(update, context, url):
    msg = await update.message.reply_text("🔗 Processing...")
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(docs)
        for t in texts: t.metadata = {"source": url}
        vector_store.add_documents(texts)
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text="✅ Done.")
    except Exception as e:
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text=f"Error: {e}")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("📥 Processing File...")
    try:
        file = await context.bot.get_file(update.message.document.file_id)
        fname = update.message.document.file_name
        with tempfile.NamedTemporaryFile(delete=True, suffix=os.path.splitext(fname)[1]) as tmp:
            await file.download_to_drive(custom_path=tmp.name)
            if fname.endswith(".pdf"): loader = PyPDFLoader(tmp.name)
            else: loader = Docx2txtLoader(tmp.name)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = splitter.split_documents(docs)
            for t in texts: t.metadata = {"source": fname}
            vector_store.add_documents(texts)
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text=f"✅ Saved: {fname}")
    except Exception as e:
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text=f"Error: {e}")

# Flask Server
flask_app = Flask('')
@flask_app.route('/')
def home(): return "Bot OK"
def run_flask(): flask_app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))

if __name__ == '__main__':
    Thread(target=run_flask).start()
    init_services()
    if TELEGRAM_BOT_TOKEN:
        app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        app.add_handler(CommandHandler('start', start))
        app.add_handler(CallbackQueryHandler(handle_callback_query))
        app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
        app.run_polling()