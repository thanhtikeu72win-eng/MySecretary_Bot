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
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legal-bot")

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

# 1. ပင်မစာမျက်နှာ (Main Menu)
MAIN_MENU = ReplyKeyboardMarkup(
    [[KeyboardButton("🧠 My Brain"), KeyboardButton("🤖 AI Assistant")]],
    resize_keyboard=True
)

# 2. AI Assistant ဝင်လိုက်ရင် ပေါ်မည့် ခလုတ်များ
AI_TOOLS_MENU = ReplyKeyboardMarkup(
    [
        [KeyboardButton("✉️ Email Draft"), KeyboardButton("📝 Summarize")],
        [KeyboardButton("🇬🇧⇄🇲🇲 Translate"), KeyboardButton("🧾 Report")],
        [KeyboardButton("🔙 Main Menu")]
    ],
    resize_keyboard=True
)

# 3. Back Button Only
BACK_BTN = ReplyKeyboardMarkup([[KeyboardButton("🔙 Back")]], resize_keyboard=True)

# ---------------------------------------------------------
# Handlers
# ---------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['mode'] = None
    context.user_data['section'] = 'main'
    await update.message.reply_text(
        "မင်္ဂလာပါ Boss! စနစ်အဆင်သင့်ဖြစ်ပါပြီ။", 
        reply_markup=MAIN_MENU
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user_mode = context.user_data.get('mode')     # Current Action (e.g., writing email)
    section = context.user_data.get('section')    # Current Menu Section (Main vs AI)

    # --- 1. Tool Execution Logic (Action Mode) ---
    if user_mode == 'add_link':
        if text.startswith("http"): await process_link(update, context, text)
        else: await update.message.reply_text("❌ Link အမှန် မဟုတ်ပါ", reply_markup=BACK_BTN)
        context.user_data['mode'] = None; return

    elif user_mode == 'delete_data':
        try:
            pinecone_index.delete(filter={"source": {"$eq": text}})
            await update.message.reply_text(f"🗑️ Deleted: {text}", reply_markup=MAIN_MENU)
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
        context.user_data['mode'] = None; context.user_data['section'] = 'main'; return

    elif user_mode in ['email', 'summarize', 'translate', 'report']:
        # Generate AI Content based on Tool
        prompt = ""
        if user_mode == 'email': prompt = f"Write a professional email about: '{text}'."
        elif user_mode == 'summarize': prompt = f"Summarize this text in bullet points: '{text}'."
        elif user_mode == 'translate': prompt = f"Translate this text (English<->Burmese): '{text}'."
        elif user_mode == 'report': prompt = f"Write a formal report about: '{text}'."
        
        await call_ai_direct(update, prompt)
        context.user_data['mode'] = None # Reset action
        return

    # --- 2. Navigation & Menu Logic ---
    
    # 2.1 Main Menu Selection
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
        await update.message.reply_text(
            "🤖 **AI Assistant Mode**\n\n- မေးခွန်းမေးလျှင် Database မှ ဖြေပါမည်။\n- Tools သုံးလိုပါက အောက်ပါခလုတ်များကို နှိပ်ပါ။", 
            reply_markup=AI_TOOLS_MENU
        )
        return

    elif text == "🔙 Main Menu":
        context.user_data['section'] = 'main'
        context.user_data['mode'] = None
        await update.message.reply_text("🔙 Main Menu သို့ ပြန်ရောက်ပါပြီ။", reply_markup=MAIN_MENU)
        return
        
    elif text == "🔙 Back":
        # Return to AI Tools Menu if came from a tool
        if section == 'ai_assistant':
            context.user_data['mode'] = None
            await update.message.reply_text("🤖 AI Assistant Mode", reply_markup=AI_TOOLS_MENU)
        else:
            await update.message.reply_text("Main Menu", reply_markup=MAIN_MENU)
        return

    # 2.2 AI Tool Selection (Inside AI Assistant)
    if section == 'ai_assistant':
        if text == "✉️ Email Draft":
            context.user_data['mode'] = 'email'
            await update.message.reply_text("✉️ Email အကြောင်းအရာကို ရေးပေးပါ Boss။", reply_markup=BACK_BTN)
            return
        elif text == "📝 Summarize":
            context.user_data['mode'] = 'summarize'
            await update.message.reply_text("📝 အကျဉ်းချုပ်လိုသော စာကို ပို့ပါ Boss။", reply_markup=BACK_BTN)
            return
        elif text == "🇬🇧⇄🇲🇲 Translate":
            context.user_data['mode'] = 'translate'
            await update.message.reply_text("🇬🇧⇄🇲🇲 ဘာသာပြန်လိုသော စာကို ပို့ပါ Boss။", reply_markup=BACK_BTN)
            return
        elif text == "🧾 Report":
            context.user_data['mode'] = 'report'
            await update.message.reply_text("🧾 Report ခေါင်းစဉ်ကို ပို့ပါ Boss။", reply_markup=BACK_BTN)
            return

    # --- 3. Default Chat (RAG) ---
    # If in AI Assistant mode and NO tool is selected, assume it's a question for Pinecone
    if section == 'ai_assistant' and not user_mode:
        if not vector_store:
            await update.message.reply_text("⚠️ Database Error")
            return
        
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        try:
            # RAG Search
            docs = vector_store.similarity_search(text, k=3)
            context_str = "\n\n".join([d.page_content for d in docs])
            
            prompt = f"Answer based on this context:\n{context_str}\n\nQuestion: {text}\n\n(Answer in Burmese)"
            response = llm.invoke(prompt)
            await update.message.reply_text(response.content, reply_markup=AI_TOOLS_MENU)
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")
        return

    # If in Main Menu and types text
    await update.message.reply_text("စကားပြောရန် သို့မဟုတ် Tools သုံးရန် '🤖 AI Assistant' ကို နှိပ်ပါ Boss။", reply_markup=MAIN_MENU)

# Helper for Direct AI Calls (Tools)
async def call_ai_direct(update, prompt_text):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        response = llm.invoke(prompt_text)
        await update.message.reply_text(response.content, reply_markup=AI_TOOLS_MENU)
    except Exception as e:
        await update.message.reply_text(f"Error: {e}", reply_markup=AI_TOOLS_MENU)

# --- Callbacks ---
async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == "add_doc": await query.edit_message_text("📥 PDF/Word ပို့ပါ Boss။")
    elif query.data == "add_link": 
        context.user_data['mode'] = 'add_link'
        await query.edit_message_text("🔗 Link ပို့ပါ Boss။")
    elif query.data == "del_data":
        context.user_data['mode'] = 'delete_data'
        await query.edit_message_text("🗑️ ဖျက်လိုသော Source Path အပြည့်အစုံပို့ပါ (Logs မှကြည့်ပါ)။")
    elif query.data == "list_mem":
        stats = pinecone_index.describe_index_stats()
        await query.edit_message_text(f"📊 Stats:\nVectors: {stats.get('total_vector_count')}")

# --- Document & Link Handling (Standard) ---
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