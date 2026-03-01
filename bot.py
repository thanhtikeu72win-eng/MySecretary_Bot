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

# ✅ MARKET SETTINGS (Default based on your screenshot)
MARKET_RATE_USD = 4500 

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

def get_currency_card_data():
    try:
        # Get Official Rates for reference
        cbm = requests.get("https://forex.cbm.gov.mm/api/latest").json()
        cbm_rates = cbm['rates']
        usd_official = float(cbm_rates['USD'].replace(',', ''))
        
        # --- Market Logic ---
        # Factor to scale other currencies based on USD Market Rate
        factor = MARKET_RATE_USD / usd_official
        
        # Sell Rates
        sell_usd = MARKET_RATE_USD
        sell_eur = float(cbm_rates['EUR'].replace(',', '')) * factor
        sell_sgd = float(cbm_rates['SGD'].replace(',', '')) * factor
        sell_thb = float(cbm_rates['THB'].replace(',', '')) * factor
        
        # Buy Rates (Spread)
        buy_usd = sell_usd - 50   # 50 kyat spread
        buy_sgd = sell_sgd - 40
        buy_thb = sell_thb - 3
        buy_eur = sell_eur - 60

        # Gold Price Logic (Estimate based on USD ratio from screenshot ~1390)
        # Screenshot: USD 4500 -> Gold 6,250,000
        gold_ratio = 1389 
        gold_high = int(sell_usd * gold_ratio)
        gold_std = int(gold_high * 0.928) # 15 P E is roughly 92-93% of High P E

        return {
            "date": cbm['info'],
            "usd": {"b": int(buy_usd), "s": int(sell_usd)},
            "sgd": {"b": int(buy_sgd), "s": int(sell_sgd)},
            "thb": {"b": int(buy_thb), "s": int(sell_thb)},
            "eur": {"b": int(buy_eur), "s": int(sell_eur)},
            "gold": {"high": gold_high, "std": gold_std}
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
    await update.message.reply_text("မင်္ဂလာပါ Boss! ရှင့်ရဲ့ အတွင်းရေးမှူးမလေး အဆင်သင့်ရှိနေပါတယ်ရှင်။ 👩‍💼\n\nဒီနေ့ ဘာကူညီပေးရမလဲ?", reply_markup=MAIN_MENU)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = update.message.text
        user_mode = context.user_data.get('mode')
        section = context.user_data.get('section', 'main')
        persona = context.user_data.get('persona', 'cute')

        # --- 1. Global Back Button ---
        if text == "🔙 Back" or text == "🔙 Main Menu":
            context.user_data['mode'] = None
            if section == 'settings':
                context.user_data['section'] = 'utils'
                await update.message.reply_text("Utilities Menu လေး ပြန်ရောက်ပါပြီရှင်။", reply_markup=UTILS_MENU)
            elif section == 'utils' or section == 'schedule' or section == 'ai_assistant':
                context.user_data['section'] = 'main'
                await update.message.reply_text("Main Menu ကို ပြန်ရောက်ပါပြီ Boss။", reply_markup=MAIN_MENU)
            else:
                context.user_data['section'] = 'main'
                await update.message.reply_text("Main Menu ပါရှင်။", reply_markup=MAIN_MENU)
            return

        # --- 2. Action Modes ---
        if user_mode == 'set_market_rate':
            global MARKET_RATE_USD
            if text.isdigit():
                MARKET_RATE_USD = int(text)
                await update.message.reply_text(f"✅ ဈေးနှုန်း ပြင်ဆင်တာ အောင်မြင်ပါတယ် Boss။\nMarket Rate (USD) = {MARKET_RATE_USD} MMK", reply_markup=SETTINGS_MENU)
            else:
                await update.message.reply_text("❌ ဂဏန်းသီးသန့်ပဲ ရိုက်ထည့်ပေးပါနော် Boss။ (ဥပမာ: 4500)", reply_markup=SETTINGS_MENU)
            context.user_data['mode'] = None; return

        elif user_mode == 'check_weather':
            city = text
            await update.message.reply_text(f"🔍 {city} မြို့အတွက် Widget လေး ထုတ်ပေးနေပါတယ်ရှင်...", reply_markup=UTILS_MENU)
            try:
                # Use wttr.in to generate a beautiful PNG Widget
                # format: _ (underscore) style, m (metric), Q (quiet), n (narrow)
                image_url = f"https://wttr.in/{city}_2mQn_lang=en.png"
                
                # Use AI for a sweet caption
                try:
                    prompt = f"Write a very short, cute weather advice for {city} in Burmese."
                    advice = llm.invoke(prompt).content
                except: advice = "ရာသီဥတု ဂရုစိုက်ပါနော် Boss။"

                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=image_url,
                    caption=f"🌤️ **Weather Widget: {city}**\n\n💡 {advice}",
                    parse_mode="Markdown",
                    reply_markup=UTILS_MENU
                )
            except:
                await update.message.reply_text("❌ မြို့နာမည် မှားနေလို့ English လို ပြန်ရိုက်ပေးပါနော်။", reply_markup=UTILS_MENU)
            context.user_data['mode'] = None; return

        elif user_mode == 'add_task':
            tasks = context.user_data.get('tasks', []); tasks.append(text); context.user_data['tasks'] = tasks
            await update.message.reply_text("✅ မှတ်သားလိုက်ပါပြီ Boss။", reply_markup=SCHEDULE_MENU); context.user_data['mode'] = None; return
        elif user_mode == 'remove_task':
            tasks = context.user_data.get('tasks', [])
            if text.isdigit() and 1 <= int(text) <= len(tasks):
                removed = tasks.pop(int(text)-1); context.user_data['tasks'] = tasks
                await update.message.reply_text(f"✅ စာရင်းမှ ပယ်ဖျက်လိုက်ပါပြီရှင်။", reply_markup=SCHEDULE_MENU)
            else: await update.message.reply_text("❌ နံပါတ် မှားနေပါတယ်ရှင်။", reply_markup=SCHEDULE_MENU)
            context.user_data['mode'] = None; return
        elif user_mode in ['email', 'summarize', 'translate', 'report']:
            await call_ai_direct(update, context, f"Task: {user_mode}. Content: {text}")
            context.user_data['mode'] = None; return

        # --- 3. Menu Navigation ---
        
        # Main Menu
        if text == "🧠 My Brain":
            context.user_data['section'] = 'brain'
            keyboard = [[InlineKeyboardButton("📥 Add PDF/Word", callback_data="add_doc"), InlineKeyboardButton("🔗 Add Link", callback_data="add_link")], [InlineKeyboardButton("📊 Stats", callback_data="list_mem"), InlineKeyboardButton("🗑️ Delete Data", callback_data="del_data")]]
            await update.message.reply_text("🧠 **My Brain Panel**\nမှတ်ဉာဏ်ပိုင်းဆိုင်ရာ စီမံခန့်ခွဲမှုတွေ ဒီမှာလုပ်နိုင်ပါတယ်ရှင်။", reply_markup=InlineKeyboardMarkup(keyboard)); return

        elif text == "🤖 AI Assistant":
            context.user_data['section'] = 'ai_assistant'
            await update.message.reply_text("🤖 **AI Assistant ပါရှင်**\nသိရှိလိုတာများကို မေးမြန်းနိုင်သလို၊ စာရေးခိုင်းတာတွေလည်း လုပ်ပေးနိုင်ပါတယ် Boss။", reply_markup=AI_TOOLS_MENU); return

        elif text == "📅 My Schedule":
            context.user_data['section'] = 'schedule'
            tasks = context.user_data.get('tasks', [])
            task_str = "\n".join([f"{i+1}. {t}" for i, t in enumerate(tasks)]) if tasks else "ဒီနေ့အတွက် ဘာမှမရှိသေးပါဘူးရှင်။"
            await update.message.reply_text(f"📅 **Today's Plan:**\n\n{task_str}", reply_markup=SCHEDULE_MENU); return

        elif text == "⚡ Utilities":
            context.user_data['section'] = 'utils'
            await update.message.reply_text("⚡ **Utilities**\nရာသီဥတုနဲ့ ငွေဈေးနှုန်းတွေ ကြည့်မလား Boss?", reply_markup=UTILS_MENU); return

        # Sub Menus
        if section == 'utils':
            if text == "🌦️ Weather":
                context.user_data['mode'] = 'check_weather'
                await update.message.reply_text("🌦️ ဘယ်မြို့အတွက် Widget ထုတ်ပေးရမလဲ Boss? (ဥပမာ: Yangon)", reply_markup=BACK_BTN); return
            
            elif text == "💰 Currency":
                await update.message.reply_text("💰 **ဈေးနှုန်းကတ်ပြား (Dashboard) ကို ထုတ်ပေးနေပါတယ်ရှင်...**", reply_markup=UTILS_MENU)
                data = get_currency_card_data()
                if data:
                    # Creating a Beautiful HTML Card to match Screenshot
                    msg = "<b>💎 DAILY MARKET RATES 💎</b>\n"
                    msg += f"📅 <i>{data['date']}</i>\n\n"
                    
                    msg += "<b>👑 ရွှေဈေး (Gold)</b>\n"
                    msg += f"⚱️ အခေါက်ရွှေ:  <b>{data['gold']['high']:,}</b> MMK\n"
                    msg += f"⚱️ ၁၅ ပဲရည်:   <b>{data['gold']['std']:,}</b> MMK\n\n"

                    msg += "<b>💵 ငွေလဲနှုန်း (Currency)</b>\n"
                    msg += "<pre>"
                    msg += " CODE |   BUY    |   SELL   \n"
                    msg += "------+----------+----------\n"
                    msg += f" USD  | {data['usd']['b']:<8} | {data['usd']['s']:<8}\n"
                    msg += f" SGD  | {data['sgd']['b']:<8} | {data['sgd']['s']:<8}\n"
                    msg += f" THB  | {data['thb']['b']:<8} | {data['thb']['s']:<8}\n"
                    msg += f" EUR  | {data['eur']['b']:<8} | {data['eur']['s']:<8}\n"
                    msg += "</pre>\n"
                    msg += f"💡 <i>Market Rate (Est): USD {MARKET_RATE_USD}</i>"
                    
                    await update.message.reply_text(msg, parse_mode="HTML", reply_markup=UTILS_MENU)
                else:
                    await update.message.reply_text("❌ Data ဆွဲမရဖြစ်နေပါတယ်ရှင်။", reply_markup=UTILS_MENU)
                return

            elif text == "⚙️ Settings":
                context.user_data['section'] = 'settings'
                await update.message.reply_text("⚙️ **Settings**\nလိုအပ်တာ ပြင်ဆင်နိုင်ပါတယ် Boss။", reply_markup=SETTINGS_MENU); return
            
            elif text == "ℹ️ About Secretary":
                await update.message.reply_text("ℹ️ **About:**\nကျွန်မက Boss ရဲ့ Personal Secretary Bot လေးပါရှင်။ 💖", reply_markup=UTILS_MENU); return

        if section == 'settings':
            if text == "✏️ Set Market Rate":
                context.user_data['mode'] = 'set_market_rate'
                await update.message.reply_text(f"💵 လက်ရှိ USD ပေါက်ဈေး ဘယ်လောက်ထားမလဲ Boss?\n(Current Setting: {MARKET_RATE_USD})", reply_markup=BACK_BTN)
                return
            elif text == "🔄 Change Persona":
                new_p = 'strict' if persona == 'cute' else 'cute'
                context.user_data['persona'] = new_p
                txt = "အခုကစပြီး တည်ငြိမ်တဲ့ပုံစံနဲ့ ပြောပါတော့မယ် Boss။" if new_p == 'strict' else "အခုကစပြီး ချစ်စရာကောင်းတဲ့ပုံစံနဲ့ ပြောပါတော့မယ်ရှင် 💖"
                await update.message.reply_text(txt, reply_markup=SETTINGS_MENU); return
            elif text == "🗑️ Clear Memory":
                context.user_data['tasks'] = []
                await update.message.reply_text("Task တွေကို ရှင်းလင်းလိုက်ပါပြီရှင်။", reply_markup=SETTINGS_MENU); return

        # --- AI Chat ---
        if section == 'ai_assistant' and not user_mode:
            if not vector_store: await update.message.reply_text("Database Error ပါရှင်။"); return
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            try:
                docs = vector_store.similarity_search(text, k=3)
                context_str = "\n".join([d.page_content for d in docs])
                prompt = f"Role: You are a polite and intelligent female secretary named 'May'. User is your 'Boss'.\nContext: {context_str}\n\nQuestion: {text}\n\nInstruction: Answer efficiently in Burmese using polite ending particles like 'ရှင်' (Shin)."
                response = llm.invoke(prompt)
                await update.message.reply_text(response.content)
            except Exception as e: await update.message.reply_text(f"Error ဖြစ်သွားလို့ပါရှင်: {e}")
            return
            
        await update.message.reply_text("Menu က ခလုတ်လေးတွေ ရွေးပေးပါနော် Boss။", reply_markup=MAIN_MENU)

    except Exception as e:
        logger.error(f"Global Handler Error: {e}")
        await update.message.reply_text("⚠️ Error လေးတစ်ခု ဖြစ်သွားလို့ Main Menu ကို ပြန်သွားပေးပါမယ်ရှင်။", reply_markup=MAIN_MENU)
        context.user_data['section'] = 'main'
        context.user_data['mode'] = None

async def call_ai_direct(update, context, prompt):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        full_prompt = "You are a polite female secretary. Answer this request from your Boss in Burmese: " + prompt
        response = llm.invoke(full_prompt)
        await update.message.reply_text(response.content)
    except: pass

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer()
    if query.data == "add_doc": await query.edit_message_text("📥 PDF/Word ဖိုင်လေး ပို့ပေးပါရှင်။")
    elif query.data == "add_link": context.user_data['mode'] = 'add_link'; await query.edit_message_text("🔗 Link လေး ပို့ပေးပါနော်။")
    elif query.data == "del_data": context.user_data['mode'] = 'delete_data'; await query.edit_message_text("🗑️ ဖျက်ချင်တဲ့ ဖိုင်နာမည် ပို့ပေးပါရှင်။")
    elif query.data == "list_mem": 
        stats = pinecone_index.describe_index_stats()
        await query.edit_message_text(f"📊 Memory Status:\nVectors: {stats.get('total_vector_count')}")

# ... (process_link and handle_document functions same as before) ...
async def process_link(update, context, url):
    msg = await update.message.reply_text("🔗 ဖတ်ရှုမှတ်သားနေပါတယ်ရှင်...")
    try:
        loader = WebBaseLoader(url); docs = loader.load(); splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200); texts = splitter.split_documents(docs)
        for t in texts: t.metadata = {"source": url}
        vector_store.add_documents(texts)
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text="✅ မှတ်သားပြီးပါပြီ Boss။")
    except: await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text="❌ Error ဖြစ်နေပါတယ်ရှင်။")

async def handle_document(update, context):
    msg = await update.message.reply_text("📥 ဖိုင်ကို စစ်ဆေးနေပါတယ်ရှင်...")
    try:
        file = await context.bot.get_file(update.message.document.file_id); fname = update.message.document.file_name
        with tempfile.NamedTemporaryFile(delete=True, suffix=os.path.splitext(fname)[1]) as tmp:
            await file.download_to_drive(custom_path=tmp.name)
            loader = PyPDFLoader(tmp.name) if fname.endswith(".pdf") else Docx2txtLoader(tmp.name)
            texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
            for t in texts: t.metadata = {"source": fname}
            vector_store.add_documents(texts)
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text=f"✅ '{fname}' ကို မှတ်ဉာဏ်ထဲ ထည့်လိုက်ပါပြီရှင်။")
    except: await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text="❌ Error ဖြစ်နေပါတယ်ရှင်။")

# Flask & Main
flask_app = Flask(''); 
@flask_app.route('/') 
def home(): return "Bot Online"
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