import os
import logging
import tempfile
import requests  # New Import for API Calls
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
# Keyboards (ခလုတ်များ)
# ---------------------------------------------------------

MAIN_MENU = ReplyKeyboardMarkup(
    [
        [KeyboardButton("🧠 My Brain"), KeyboardButton("🤖 AI Assistant")],
        [KeyboardButton("📅 My Schedule"), KeyboardButton("⚡ Utilities")]
    ], resize_keyboard=True
)

AI_TOOLS_MENU = ReplyKeyboardMarkup(
    [
        [KeyboardButton("✉️ Email Draft"), KeyboardButton("📝 Summarize")],
        [KeyboardButton("🇬🇧⇄🇲🇲 Translate"), KeyboardButton("🧾 Report")],
        [KeyboardButton("🔙 Main Menu")]
    ], resize_keyboard=True
)

SCHEDULE_MENU = ReplyKeyboardMarkup(
    [
        [KeyboardButton("➕ Reminder သစ်"), KeyboardButton("📋 စာရင်းကြည့်")],
        [KeyboardButton("✅ Task Done"), KeyboardButton("🔙 Main Menu")]
    ], resize_keyboard=True
)

UTILS_MENU = ReplyKeyboardMarkup(
    [
        [KeyboardButton("🌦️ Weather"), KeyboardButton("💰 Currency")],
        [KeyboardButton("⚙️ Settings"), KeyboardButton("ℹ️ About Secretary")],
        [KeyboardButton("🔙 Main Menu")]
    ], resize_keyboard=True
)

SETTINGS_MENU = ReplyKeyboardMarkup(
    [
        [KeyboardButton("🔄 Change Persona"), KeyboardButton("🗑️ Clear Memory")],
        [KeyboardButton("🔙 Back")]
    ], resize_keyboard=True
)

BACK_BTN = ReplyKeyboardMarkup([[KeyboardButton("🔙 Back")]], resize_keyboard=True)

# ---------------------------------------------------------
# Handlers
# ---------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['mode'] = None
    context.user_data['section'] = 'main'
    # Default Persona
    if 'persona' not in context.user_data:
        context.user_data['persona'] = 'cute' # Default: Cute
        
    await update.message.reply_text(
        "မင်္ဂလာပါ Boss! ရှင့်ရဲ့ အတွင်းရေးမှူးမလေး အဆင်သင့်ရှိနေပါတယ်ရှင်။\n\nဒီနေ့ ဘာဆောင်ရွက်ပေးရမလဲ?", 
        reply_markup=MAIN_MENU
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user_mode = context.user_data.get('mode')
    section = context.user_data.get('section')
    persona = context.user_data.get('persona', 'cute') # Get current persona

    # --- 1. Action Modes (Input လက်ခံနေချိန်) ---
    if user_mode == 'add_link':
        if text.startswith("http"): await process_link(update, context, text)
        else: await update.message.reply_text("❌ Link အမှန် မဟုတ်ပါရှင်", reply_markup=BACK_BTN)
        context.user_data['mode'] = None; return

    elif user_mode == 'delete_data':
        try:
            pinecone_index.delete(filter={"source": {"$eq": text}})
            await update.message.reply_text(f"🗑️ ဖိုင်ကို ဖျက်လိုက်ပါပြီ Boss: {text}", reply_markup=MAIN_MENU)
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
        context.user_data['mode'] = None; context.user_data['section'] = 'main'; return

    # --- 📅 Schedule Actions ---
    elif user_mode == 'add_task':
        tasks = context.user_data.get('tasks', [])
        tasks.append(text)
        context.user_data['tasks'] = tasks
        await update.message.reply_text(f"✅ မှတ်သားလိုက်ပါပြီ Boss: '{text}'", reply_markup=SCHEDULE_MENU)
        context.user_data['mode'] = None; return
    
    elif user_mode == 'remove_task':
        tasks = context.user_data.get('tasks', [])
        if text.isdigit() and 1 <= int(text) <= len(tasks):
            removed = tasks.pop(int(text)-1)
            context.user_data['tasks'] = tasks
            await update.message.reply_text(f"✅ ပြီးစီးကြောင်း မှတ်လိုက်ပါပြီ: '{removed}'", reply_markup=SCHEDULE_MENU)
        else:
            await update.message.reply_text("❌ နံပါတ်မှားနေပါတယ်ရှင်။ ပြန်ရိုက်ပေးပါနော်။", reply_markup=SCHEDULE_MENU)
        context.user_data['mode'] = None; return

    # --- 🌦️ Weather Action ---
    elif user_mode == 'check_weather':
        city = text
        await update.message.reply_text(f"🔍 {city} မြို့ရဲ့ ရာသီဥတုကို ရှာဖွေနေပါတယ်ရှင်...")
        try:
            # Using wttr.in for weather (No API Key needed)
            url = f"https://wttr.in/{city}?format=%C+%t+(%f)+%w"
            response = requests.get(url)
            if response.status_code == 200:
                weather_info = response.text.strip()
                await update.message.reply_text(f"🌤️ **Weather Report for {city}:**\n{weather_info}", reply_markup=UTILS_MENU)
            else:
                await update.message.reply_text("❌ မြို့နာမည် မှားနေပုံရပါတယ်ရှင်။", reply_markup=UTILS_MENU)
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}", reply_markup=UTILS_MENU)
        context.user_data['mode'] = None; return

    # --- 🤖 AI Tool Actions ---
    elif user_mode in ['email', 'summarize', 'translate', 'report']:
        # Persona Logic in Prompt
        tone = "polite, cute and helpful female secretary" if persona == 'cute' else "formal, strict and professional assistant"
        
        prompt = ""
        if user_mode == 'email': prompt = f"You are a {tone}. Draft a professional email about: '{text}'."
        elif user_mode == 'summarize': prompt = f"Summarize this text in bullet points: '{text}'."
        elif user_mode == 'translate': prompt = f"Translate this text (English<->Burmese): '{text}'."
        elif user_mode == 'report': prompt = f"Write a formal report about: '{text}'."
        
        await call_ai_direct(update, context, prompt) # Fixed: Passed context
        context.user_data['mode'] = None; return

    # --- 2. Menu Navigation Logic ---
    
    if text == "🧠 My Brain":
        context.user_data['section'] = 'brain'
        keyboard = [
            [InlineKeyboardButton("📥 Add PDF/Word", callback_data="add_doc"), InlineKeyboardButton("🔗 Add Link", callback_data="add_link")],
            [InlineKeyboardButton("📊 Stats", callback_data="list_mem"), InlineKeyboardButton("🗑️ Delete Data", callback_data="del_data")]
        ]
        await update.message.reply_text("🧠 **My Brain Panel:**", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
        return

    elif text == "🤖 AI Assistant":
        context.user_data['section'] = 'ai_assistant'
        await update.message.reply_text("🤖 **AI Assistant Mode**\nမေးခွန်းမေးလျှင် Database မှ ဖြေပါမည်။", reply_markup=AI_TOOLS_MENU)
        return

    elif text == "📅 My Schedule":
        context.user_data['section'] = 'schedule'
        tasks = context.user_data.get('tasks', [])
        task_str = "\n".join([f"{i+1}. {t}" for i, t in enumerate(tasks)]) if tasks else "ဘာမှမရှိသေးပါဘူးရှင်။"
        await update.message.reply_text(f"📅 **Today's Plan:**\n\n{task_str}", reply_markup=SCHEDULE_MENU)
        return

    elif text == "⚡ Utilities":
        context.user_data['section'] = 'utils'
        await update.message.reply_text("⚡ **Utilities**", reply_markup=UTILS_MENU)
        return

    elif text == "🔙 Main Menu":
        context.user_data['section'] = 'main'
        context.user_data['mode'] = None
        await update.message.reply_text("🔙 Main Menu သို့ ပြန်ရောက်ပါပြီ Boss။", reply_markup=MAIN_MENU)
        return

    elif text == "🔙 Back":
        if section == 'ai_assistant': await update.message.reply_text("🤖 AI Mode", reply_markup=AI_TOOLS_MENU)
        elif section == 'schedule': await update.message.reply_text("📅 Schedule Mode", reply_markup=SCHEDULE_MENU)
        elif section == 'utils': await update.message.reply_text("⚡ Utilities Mode", reply_markup=UTILS_MENU)
        elif section == 'settings': await update.message.reply_text("⚙️ Settings Mode", reply_markup=SETTINGS_MENU)
        else: await update.message.reply_text("Main Menu", reply_markup=MAIN_MENU)
        return

    # --- 3. Sub-Menu Features ---

    # 3.1 Schedule Features
    if section == 'schedule':
        if text == "➕ Reminder သစ်":
            context.user_data['mode'] = 'add_task'
            await update.message.reply_text("📝 ဘာမှတ်ထားချင်လဲ ပြောပါ Boss။", reply_markup=BACK_BTN)
            return
        elif text == "📋 စာရင်းကြည့်":
            tasks = context.user_data.get('tasks', [])
            task_str = "\n".join([f"{i+1}. {t}" for i, t in enumerate(tasks)]) if tasks else "Empty"
            await update.message.reply_text(f"📋 **Task List:**\n{task_str}", reply_markup=SCHEDULE_MENU)
            return
        elif text == "✅ Task Done":
            tasks = context.user_data.get('tasks', [])
            if not tasks:
                await update.message.reply_text("ပြီးစရာ Task မရှိသေးပါဘူးရှင်။", reply_markup=SCHEDULE_MENU)
            else:
                context.user_data['mode'] = 'remove_task'
                task_str = "\n".join([f"{i+1}. {t}" for i, t in enumerate(tasks)])
                await update.message.reply_text(f"✅ ဘယ်နံပါတ် ပြီးသွားပြီလဲရှင်?\n\n{task_str}\n\n(နံပါတ်ပဲ ရိုက်ထည့်ပေးပါ)", reply_markup=BACK_BTN)
            return

    # 3.2 Utilities Features
    if section == 'utils':
        if text == "🌦️ Weather":
            context.user_data['mode'] = 'check_weather'
            await update.message.reply_text("🌦️ ဘယ်မြို့အတွက် သိချင်လဲရှင့်? (ဥပမာ: Yangon, Mandalay)", reply_markup=BACK_BTN)
            return
        
        elif text == "💰 Currency":
            await update.message.reply_text("💰 **Central Bank Exchange Rates:**\nဈေးနှုန်းများကို ဆွဲယူနေပါတယ်ရှင်...", reply_markup=UTILS_MENU)
            try:
                # CBM Official API
                res = requests.get("https://forex.cbm.gov.mm/api/latest").json()
                rates = res.get('rates', {})
                date = res.get('info', 'Today')
                
                msg = f"📅 **Date:** {date}\n\n"
                msg += f"🇺🇸 **USD:** {rates.get('USD', 'N/A')} MMK\n"
                msg += f"🇪🇺 **EUR:** {rates.get('EUR', 'N/A')} MMK\n"
                msg += f"🇸🇬 **SGD:** {rates.get('SGD', 'N/A')} MMK\n"
                msg += f"🇹🇭 **THB:** {rates.get('THB', 'N/A')} MMK\n"
                msg += "\n(Source: Central Bank of Myanmar)"
                
                await update.message.reply_text(msg, reply_markup=UTILS_MENU)
            except Exception as e:
                await update.message.reply_text(f"❌ Error fetching rates: {e}", reply_markup=UTILS_MENU)
            return
        
        elif text == "⚙️ Settings":
            context.user_data['section'] = 'settings'
            p_name = "Cute/Friendly" if persona == 'cute' else "Strict/Professional"
            await update.message.reply_text(f"⚙️ **Settings Panel**\nCurrent Persona: {p_name}", reply_markup=SETTINGS_MENU)
            return

        elif text == "ℹ️ About Secretary":
            about_msg = """
ℹ️ **About Your Secretary Bot** 👩‍💼

ကျွန်မက Boss ရဲ့ ကိုယ်ပိုင် Digital အတွင်းရေးမှူးမလေး ဖြစ်ပါတယ်ရှင်။
ကျွန်မ လုပ်ပေးနိုင်တာတွေကတော့ -

1.  **🧠 My Brain:** စာရွက်စာတမ်း (PDF/Word) တွေကို ဖတ်ပြီး မှတ်ထားပေးပါတယ်။ မေးသမျှကို ပြန်ဖြေပေးပါတယ်။
2.  **📅 My Schedule:** နေ့စဉ် လုပ်စရာတွေကို မှတ်ပေး၊ သတိပေးပါတယ်။
3.  **🌦️ Weather:** မိုးလေဝသ အခြေအနေကို အချိန်နဲ့တပြေးညီ ကြည့်ပေးပါတယ်။
4.  **💰 Currency:** ဗဟိုဘဏ် ပေါက်ဈေးတွေကို ကြည့်ပေးပါတယ်။
5.  **🤖 AI Tools:** Email ရေးခြင်း၊ ဘာသာပြန်ခြင်း၊ Report ရေးခြင်းတို့ကို ကူညီပေးပါတယ်။

Boss စိတ်တိုင်းကျ ခိုင်းစေနိုင်ပါတယ်ရှင်! 💖
            """
            await update.message.reply_text(about_msg, reply_markup=UTILS_MENU)
            return

    # 3.3 Settings Features
    if section == 'settings':
        if text == "🔄 Change Persona":
            current = context.user_data.get('persona', 'cute')
            new_persona = 'strict' if current == 'cute' else 'cute'
            context.user_data['persona'] = new_persona
            
            msg = "👩‍💼 **Persona Updated:** Now Strict & Professional." if new_persona == 'strict' else "👩‍💼 **Persona Updated:** Now Cute & Friendly! 💖"
            await update.message.reply_text(msg, reply_markup=SETTINGS_MENU)
            return
            
        elif text == "🗑️ Clear Memory":
            context.user_data['tasks'] = []
            await update.message.reply_text("🗑️ Schedule များကို ရှင်းလင်းလိုက်ပါပြီ Boss။", reply_markup=SETTINGS_MENU)
            return

    # 3.4 AI Tools
    if section == 'ai_assistant':
        if text == "✉️ Email Draft": context.user_data['mode'] = 'email'; await update.message.reply_text("✉️ Email Topic ပြောပေးပါရှင်။", reply_markup=BACK_BTN); return
        elif text == "📝 Summarize": context.user_data['mode'] = 'summarize'; await update.message.reply_text("📝 စာသား ပို့ပေးပါရှင်။", reply_markup=BACK_BTN); return
        elif text == "🇬🇧⇄🇲🇲 Translate": context.user_data['mode'] = 'translate'; await update.message.reply_text("🇬🇧⇄🇲🇲 ဘာသာပြန်လိုသော စာသား ပို့ပေးပါရှင်။", reply_markup=BACK_BTN); return
        elif text == "🧾 Report": context.user_data['mode'] = 'report'; await update.message.reply_text("🧾 Report ခေါင်းစဉ် ပြောပေးပါရှင်။", reply_markup=BACK_BTN); return

    # --- 4. Default RAG Chat ---
    if section == 'ai_assistant' and not user_mode:
        if not vector_store: await update.message.reply_text("⚠️ Database Error"); return
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        try:
            docs = vector_store.similarity_search(text, k=3)
            context_str = "\n".join([d.page_content for d in docs])
            
            tone_instruct = "Answer efficiently and professionally." if persona == 'strict' else "Answer politely and helpfully with a secretary tone (use 'ရှင်')."
            
            prompt = f"Context:\n{context_str}\n\nQuestion: {text}\n\nInstruction: {tone_instruct} Answer in Burmese."
            response = llm.invoke(prompt)
            await update.message.reply_text(response.content, reply_markup=AI_TOOLS_MENU)
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")
        return

    # Fallback
    await update.message.reply_text("တခုခု ခိုင်းစေလိုရင် Menu ကနေ ရွေးပေးပါ Boss။", reply_markup=MAIN_MENU)

# Helper for Direct AI Calls
async def call_ai_direct(update, context, prompt):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        response = llm.invoke(prompt)
        await update.message.reply_text(response.content)
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

# --- Callbacks ---
async def handle_callback_query(update, context):
    query = update.callback_query; await query.answer()
    if query.data == "add_doc": await query.edit_message_text("📥 PDF/Word ပို့ပေးပါရှင်။")
    elif query.data == "add_link": context.user_data['mode'] = 'add_link'; await query.edit_message_text("🔗 Link ပို့ပေးပါရှင်။")
    elif query.data == "del_data": context.user_data['mode'] = 'delete_data'; await query.edit_message_text("🗑️ ဖျက်လိုသော Path အပြည့်အစုံ ပို့ပေးပါရှင်။")
    elif query.data == "list_mem": 
        stats = pinecone_index.describe_index_stats()
        await query.edit_message_text(f"📊 Memory: {stats.get('total_vector_count')} Items")

# --- Doc/Link Processing ---
async def process_link(update, context, url):
    msg = await update.message.reply_text("🔗 ဖတ်နေပါတယ်ရှင်...")
    try:
        loader = WebBaseLoader(url); docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200); texts = splitter.split_documents(docs)
        for t in texts: t.metadata = {"source": url}
        vector_store.add_documents(texts)
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text="✅ မှတ်သားပြီးပါပြီရှင်။")
    except Exception as e: await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text=f"Error: {e}")

async def handle_document(update, context):
    msg = await update.message.reply_text("📥 ဖိုင်ကို ဖတ်နေပါတယ်ရှင်...")
    try:
        file = await context.bot.get_file(update.message.document.file_id); fname = update.message.document.file_name
        with tempfile.NamedTemporaryFile(delete=True, suffix=os.path.splitext(fname)[1]) as tmp:
            await file.download_to_drive(custom_path=tmp.name)
            loader = PyPDFLoader(tmp.name) if fname.endswith(".pdf") else Docx2txtLoader(tmp.name)
            texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
            for t in texts: t.metadata = {"source": fname}
            vector_store.add_documents(texts)
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text=f"✅ '{fname}' ကို မှတ်သားပြီးပါပြီရှင်။")
    except Exception as e: await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text=f"Error: {e}")

# Flask & Main
flask_app = Flask(''); 
@flask_app.route('/') 
def home(): return "Secretary Bot Online"
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