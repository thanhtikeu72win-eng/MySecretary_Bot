import os
import logging
import tempfile
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
            # Persona Prompt Injection
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

        if PINECONE_API_KEY and GOOGLE_API_KEY:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            pinecone_index = pc.Index(PINECONE_INDEX_NAME)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)
            vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings)
            logger.info("✅ Secretary Brain Initialized")
    except Exception as e:
        logger.error(f"❌ Service Init Error: {e}")

# ---------------------------------------------------------
# Keyboards (ခလုတ်များ) 🎛️
# ---------------------------------------------------------

# 1. Main Menu
MAIN_MENU = ReplyKeyboardMarkup(
    [
        [KeyboardButton("🧠 My Brain"), KeyboardButton("🤖 AI Assistant")],
        [KeyboardButton("📅 My Schedule"), KeyboardButton("⚡ Utilities")]
    ],
    resize_keyboard=True
)

# 2. AI Tools Menu
AI_TOOLS_MENU = ReplyKeyboardMarkup(
    [
        [KeyboardButton("✉️ Email Draft"), KeyboardButton("📝 Summarize")],
        [KeyboardButton("🇬🇧⇄🇲🇲 Translate"), KeyboardButton("🧾 Report")],
        [KeyboardButton("🔙 Main Menu")]
    ],
    resize_keyboard=True
)

# 3. Schedule Menu (New)
SCHEDULE_MENU = ReplyKeyboardMarkup(
    [
        [KeyboardButton("➕ Reminder သစ်"), KeyboardButton("📋 စာရင်းကြည့်")],
        [KeyboardButton("✅ Task Done"), KeyboardButton("🔙 Main Menu")]
    ],
    resize_keyboard=True
)

# 4. Utilities Menu (New)
UTILITY_MENU = ReplyKeyboardMarkup(
    [
        [KeyboardButton("🌤️ Weather"), KeyboardButton("💱 Currency")],
        [KeyboardButton("⚙️ Settings"), KeyboardButton("🔙 Main Menu")]
    ],
    resize_keyboard=True
)

# Back Button
BACK_BTN = ReplyKeyboardMarkup([[KeyboardButton("🔙 Back")]], resize_keyboard=True)

# ---------------------------------------------------------
# Handlers (လုပ်ဆောင်ချက်များ) 👩‍💼
# ---------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['mode'] = None
    context.user_data['section'] = 'main'
    # Initialize Task List if not exists
    if 'tasks' not in context.user_data: context.user_data['tasks'] = []
    
    await update.message.reply_text(
        "မင်္ဂလာပါ Boss ရှင်! 🙏\nကျွန်မက Boss ရဲ့ ကိုယ်ပိုင် အတွင်းရေးမှူးမလေးပါ။\nအလုပ်ကိစ္စတွေ ကူညီပေးဖို့ အဆင်သင့်ပါရှင်။", 
        reply_markup=MAIN_MENU
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user_mode = context.user_data.get('mode')
    section = context.user_data.get('section')
    tasks = context.user_data.setdefault('tasks', [])

    # --- 1. Special Input Modes (Waiting for user typing) ---
    
    # 1.1 Reminder Input
    if user_mode == 'add_reminder':
        tasks.append(text)
        await update.message.reply_text(f"✅ မှတ်သားလိုက်ပါပြီ Boss ရှင်!\n📌 Reminder: {text}", reply_markup=SCHEDULE_MENU)
        context.user_data['mode'] = None
        return

    # 1.2 Utilities Input (Simulated AI response for Weather/Currency)
    elif user_mode == 'ask_weather':
        await call_ai_direct(update, f"Tell me a short weather forecast or advice for: {text} (Keep it brief and polite as a secretary).")
        context.user_data['mode'] = None
        return
    elif user_mode == 'ask_currency':
        await call_ai_direct(update, f"Convert or give exchange rate info for: {text} (Keep it brief).")
        context.user_data['mode'] = None
        return

    # 1.3 Brain & Tool Inputs (Same as before)
    elif user_mode == 'add_link':
        if text.startswith("http"): await process_link(update, context, text)
        else: await update.message.reply_text("Link အမှန် မဟုတ်ပါဘူး Boss ရှင်..", reply_markup=BACK_BTN)
        context.user_data['mode'] = None; return

    elif user_mode == 'delete_data':
        try:
            pinecone_index.delete(filter={"source": {"$eq": text}})
            await update.message.reply_text(f"🗑️ ဖိုင်ကို ဖျက်လိုက်ပါပြီ Boss ရှင်: {text}", reply_markup=MAIN_MENU)
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
        context.user_data['mode'] = None; context.user_data['section'] = 'main'; return

    elif user_mode in ['email', 'summarize', 'translate', 'report']:
        # Persona Prompt
        prompt = ""
        if user_mode == 'email': prompt = f"You are a smart secretary. Draft a professional email about: '{text}'."
        elif user_mode == 'summarize': prompt = f"Summarize this politely: '{text}'."
        elif user_mode == 'translate': prompt = f"Translate accurately (English<->Burmese): '{text}'."
        elif user_mode == 'report': prompt = f"Write a formal report about: '{text}'."
        
        await call_ai_direct(update, prompt)
        context.user_data['mode'] = None 
        return

    # --- 2. Menu Navigation ---

    # 2.1 Main Sections
    if text == "🧠 My Brain":
        context.user_data['section'] = 'brain'
        keyboard = [
            [InlineKeyboardButton("📥 Add PDF/Word", callback_data="add_doc"), InlineKeyboardButton("🔗 Add Link", callback_data="add_link")],
            [InlineKeyboardButton("📊 Stats", callback_data="list_mem"), InlineKeyboardButton("🗑️ Delete Data", callback_data="del_data")]
        ]
        await update.message.reply_text("🧠 **My Brain Panel:**\nမှတ်ဉာဏ်တွေကို ဒီကနေ စီမံလို့ရပါတယ် Boss ရှင်။", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
        return

    elif text == "🤖 AI Assistant":
        context.user_data['section'] = 'ai_assistant'
        await update.message.reply_text("🤖 **AI Assistant Mode**\nဘာကူညီပေးရမလဲ Boss ရှင်? မေးခွန်းမေးမလား၊ Tool သုံးမလားရှင်?", reply_markup=AI_TOOLS_MENU)
        return

    elif text == "📅 My Schedule":
        context.user_data['section'] = 'schedule'
        # Check tasks
        task_msg = "📅 **Today's Plan:**\n"
        if not tasks:
            task_msg += "(လောလောဆယ် ဘာမှမရှိသေးပါဘူး Boss ရှင်)"
        else:
            for i, t in enumerate(tasks, 1):
                task_msg += f"{i}. {t}\n"
        
        await update.message.reply_text(task_msg, reply_markup=SCHEDULE_MENU, parse_mode="Markdown")
        return

    elif text == "⚡ Utilities":
        context.user_data['section'] = 'utilities'
        await update.message.reply_text("⚡ **Utilities Mode**\nအခြား ၀န်ဆောင်မှုများပါရှင်။", reply_markup=UTILITY_MENU)
        return

    elif text == "🔙 Main Menu":
        context.user_data['section'] = 'main'
        context.user_data['mode'] = None
        await update.message.reply_text("🔙 Main Menu သို့ ပြန်ရောက်ပါပြီ Boss ရှင်။", reply_markup=MAIN_MENU)
        return

    # 2.2 Sub-Menu Actions
    
    # Schedule Actions
    if section == 'schedule':
        if text == "➕ Reminder သစ်":
            context.user_data['mode'] = 'add_reminder'
            await update.message.reply_text("📝 ဘာမှတ်ထားပေးရမလဲ Boss ရှင်?", reply_markup=BACK_BTN)
            return
        elif text == "📋 စာရင်းကြည့်":
            # Just re-trigger the menu view
            task_msg = "📅 **Current Tasks:**\n" + ("\n".join([f"- {t}" for t in tasks]) if tasks else "(Empty)")
            await update.message.reply_text(task_msg, reply_markup=SCHEDULE_MENU, parse_mode="Markdown")
            return
        elif text == "✅ Task Done":
            if tasks:
                removed = tasks.pop(0) # Remove first for simplicity or clear all
                await update.message.reply_text(f"✅ '{removed}' ကို ပြီးစီးကြောင်း မှတ်သားလိုက်ပါပြီ။", reply_markup=SCHEDULE_MENU)
            else:
                await update.message.reply_text("ပြီးစရာ အလုပ်မရှိသေးပါဘူး Boss ရှင်။", reply_markup=SCHEDULE_MENU)
            return

    # Utilities Actions
    if section == 'utilities':
        if text == "🌤️ Weather":
            context.user_data['mode'] = 'ask_weather'
            await update.message.reply_text("🌤️ ဘယ်မြို့အတွက် ရာသီဥတု သိချင်လဲ ပြောပေးပါရှင်။", reply_markup=BACK_BTN)
            return
        elif text == "💱 Currency":
            context.user_data['mode'] = 'ask_currency'
            await update.message.reply_text("💱 ဘယ်ငွေကြေးကို တွက်ချင်တာလဲ ပြောပါရှင် (e.g. 100 USD to MMK)။", reply_markup=BACK_BTN)
            return
        elif text == "⚙️ Settings":
            await update.message.reply_text("⚙️ **Settings:**\nLanguage: Myanmar\nRole: Secretary\nVersion: 2.0", reply_markup=UTILITY_MENU)
            return

    # AI Tool Actions (Same as before)
    if section == 'ai_assistant':
        if text == "✉️ Email Draft": context.user_data['mode'] = 'email'; await update.message.reply_text("✉️ Email အကြောင်းအရာ ပြောပေးပါရှင်။", reply_markup=BACK_BTN); return
        elif text == "📝 Summarize": context.user_data['mode'] = 'summarize'; await update.message.reply_text("📝 စာပို့ပေးပါရှင်။", reply_markup=BACK_BTN); return
        elif text == "🇬🇧⇄🇲🇲 Translate": context.user_data['mode'] = 'translate'; await update.message.reply_text("🇬🇧⇄🇲🇲 ဘာသာပြန်လိုသော စာပို့ပါရှင်။", reply_markup=BACK_BTN); return
        elif text == "🧾 Report": context.user_data['mode'] = 'report'; await update.message.reply_text("🧾 Report ခေါင်းစဉ် ပြောပါရှင်။", reply_markup=BACK_BTN); return

    # Back Button Logic
    if text == "🔙 Back":
        if section == 'schedule': await update.message.reply_text("Schedule Menu", reply_markup=SCHEDULE_MENU)
        elif section == 'utilities': await update.message.reply_text("Utilities Menu", reply_markup=UTILITY_MENU)
        elif section == 'ai_assistant': await update.message.reply_text("AI Assistant Mode", reply_markup=AI_TOOLS_MENU)
        else: await update.message.reply_text("Main Menu", reply_markup=MAIN_MENU)
        return

    # --- 3. Default Chat (RAG) ---
    if section == 'ai_assistant' and not user_mode:
        if not vector_store: await update.message.reply_text("⚠️ Brain မချိတ်ရသေးပါဘူးရှင်။"); return
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        try:
            docs = vector_store.similarity_search(text, k=3)
            context_str = "\n\n".join([d.page_content for d in docs])
            prompt = f"Act as a smart female secretary. Answer concisely and politely in Burmese based on:\n{context_str}\n\nQuestion: {text}"
            response = llm.invoke(prompt)
            await update.message.reply_text(response.content, reply_markup=AI_TOOLS_MENU)
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")
        return

    # Fallback
    await update.message.reply_text("ဘာကူညီပေးရမလဲ Boss ရှင်? (AI Assistant ကို နှိပ်ပြီး ပြောပေးပါနော်)", reply_markup=MAIN_MENU)

# Helper for Direct AI
async def call_ai_direct(update, prompt_text):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        response = llm.invoke(prompt_text)
        # Check which menu to show back
        markup = MAIN_MENU 
        # (For simplicity return to Main or keep contextual. Let's return to Main for clean flow or stay in section if possible)
        await update.message.reply_text(response.content)
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

# Callbacks (Brain)
async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "add_doc": await query.edit_message_text("📥 ဖိုင်ပို့ပေးပါ Boss ရှင်။")
    elif query.data == "add_link": context.user_data['mode'] = 'add_link'; await query.edit_message_text("🔗 Link ပို့ပေးပါရှင်။")
    elif query.data == "del_data": context.user_data['mode'] = 'delete_data'; await query.edit_message_text("🗑️ ဖျက်ချင်တဲ့ File Path ပို့ပေးပါရှင်။")
    elif query.data == "list_mem": 
        stats = pinecone_index.describe_index_stats()
        await query.edit_message_text(f"📊 မှတ်ဉာဏ်အခြေအနေ:\nVectors: {stats.get('total_vector_count')}")

# Doc & Link Processing (Same logic)
async def process_link(update, context, url):
    msg = await update.message.reply_text("🔗 သိမ်းဆည်းနေပါတယ်ရှင်...")
    try:
        loader = WebBaseLoader(url); docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(docs)
        for t in texts: t.metadata = {"source": url}
        vector_store.add_documents(texts)
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text="✅ သိမ်းပြီးပါပြီရှင်။")
    except Exception as e: await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text=f"Error: {e}")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("📥 ဖတ်နေပါတယ်ရှင်...")
    try:
        file = await context.bot.get_file(update.message.document.file_id)
        fname = update.message.document.file_name
        with tempfile.NamedTemporaryFile(delete=True, suffix=os.path.splitext(fname)[1]) as tmp:
            await file.download_to_drive(custom_path=tmp.name)
            if fname.endswith(".pdf"): loader = PyPDFLoader(tmp.name)
            else: loader = Docx2txtLoader(tmp.name)
            docs = loader.load(); splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = splitter.split_documents(docs)
            for t in texts: t.metadata = {"source": fname}
            vector_store.add_documents(texts)
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text=f"✅ ဖိုင်သိမ်းပြီးပါပြီ Boss ရှင်: {fname}")
    except Exception as e: await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text=f"Error: {e}")

# Server
flask_app = Flask('')
@flask_app.route('/')
def home(): return "Secretary Bot Ready"
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