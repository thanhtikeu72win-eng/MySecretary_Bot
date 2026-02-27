import os
import logging
import asyncio
import tempfile
from flask import Flask
from threading import Thread
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, CallbackQueryHandler, filters

# Gemini & LangChain Imports
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client

# 1. Setup Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 2. Load Environment Variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Debug Check
print(f"DEBUG CHECK: TELEGRAM_BOT_TOKEN is {'✅ OK' if TELEGRAM_BOT_TOKEN else '❌ MISSING'}")
print(f"DEBUG CHECK: GOOGLE_API_KEY is {'✅ OK' if GOOGLE_API_KEY else '❌ MISSING'}")
print(f"DEBUG CHECK: SUPABASE_URL is {'✅ OK' if SUPABASE_URL else '❌ MISSING'}")
print(f"DEBUG CHECK: SUPABASE_KEY is {'✅ OK' if SUPABASE_KEY else '❌ MISSING'}")

# 3. Global Variables (Initialize as None)
vector_store = None
llm = None

# ---------------------------------------------------------
# Initialization Function (Safe Mode)
# ---------------------------------------------------------
def init_services():
    global vector_store, llm
    try:
        # Init Gemini
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.7
            )
            logger.info("✅ Gemini Model Initialized")

        # Init Supabase (Potential Crash Point - Wrapped in Try/Except)
        if SUPABASE_URL and SUPABASE_KEY:
            supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
            
            vector_store = SupabaseVectorStore(
                client=supabase,
                embedding=embeddings,
                table_name="documents",
                query_name="match_documents"
            )
            logger.info("✅ Supabase Vector Store Initialized")
        else:
            logger.warning("⚠️ Supabase Credentials Missing")
            
    except Exception as e:
        logger.error(f"❌ Service Initialization Error: {e}")
        # We DO NOT exit here. We let the bot run without DB.

# ---------------------------------------------------------
# UI Layout
# ---------------------------------------------------------
MAIN_MENU_KEYBOARD = ReplyKeyboardMarkup(
    [
        ["🧠 My Brain", "🤖 AI Assistant"],
        ["📅 My Schedule", "⚡ Utilities"]
    ],
    resize_keyboard=True,
    one_time_keyboard=False
)

# ---------------------------------------------------------
# Handlers
# ---------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_msg = "မင်္ဂလာပါ Boss! Bot အလုပ်လုပ်နေပါပြီ။"
    if vector_store is None:
        welcome_msg += "\n⚠️ သတိပေးချက်: Database မချိတ်ဆက်ထားပါ။ (My Brain သုံးမရပါ)"
    
    await update.message.reply_text(welcome_msg, reply_markup=MAIN_MENU_KEYBOARD)

async def handle_menu_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    
    if text == "🧠 My Brain":
        keyboard = [
            [InlineKeyboardButton("📥 Add PDF", callback_data="add_pdf"), InlineKeyboardButton("🔗 Add Link", callback_data="add_link")],
            [InlineKeyboardButton("🧹 Clear Memory", callback_data="clear_memory")]
        ]
        await update.message.reply_text("🧠 My Brain Panel:", reply_markup=InlineKeyboardMarkup(keyboard))
        
    elif text == "🤖 AI Assistant":
        await update.message.reply_text("🤖 AI Feature coming next.")
    
    else:
        # Chat Logic
        await update.message.reply_chat_action("typing")
        
        if llm:
            try:
                # Try RAG first if DB is active
                context_text = ""
                if vector_store:
                    try:
                        docs = vector_store.similarity_search(text, k=2)
                        context_text = "\n".join([d.page_content for d in docs])
                    except Exception as db_err:
                        logger.error(f"DB Search Error: {db_err}")

                prompt = f"Answer this: {text}"
                if context_text:
                    prompt = f"Context: {context_text}\n\nQuestion: {text}"
                
                response = llm.invoke(prompt)
                await update.message.reply_text(response.content, reply_markup=MAIN_MENU_KEYBOARD)
            except Exception as e:
                await update.message.reply_text(f"❌ AI Error: {e}")
        else:
            await update.message.reply_text("❌ AI Model not loaded.")

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "add_pdf":
        await query.edit_message_text("📥 PDF ဖိုင် ပို့ပေးပါ Boss။")

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not vector_store:
        await update.message.reply_text("❌ Database Error: My Brain မရှိသေးပါ။", reply_markup=MAIN_MENU_KEYBOARD)
        return

    document = update.message.document
    if document.mime_type != 'application/pdf':
        await update.message.reply_text("PDF Only!", reply_markup=MAIN_MENU_KEYBOARD)
        return

    status = await update.message.reply_text("📥 Saving...", reply_markup=MAIN_MENU_KEYBOARD)
    try:
        file = await context.bot.get_file(document.file_id)
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
            await file.download_to_drive(custom_path=temp_pdf.name)
            loader = PyPDFLoader(temp_pdf.name)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(pages)
            for doc in texts: doc.metadata = {"source": document.file_name}
            
            vector_store.add_documents(texts)
            
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status.message_id, text=f"✅ Saved: {document.file_name}")
    except Exception as e:
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status.message_id, text=f"❌ Error: {e}")

# ---------------------------------------------------------
# Main Execution (Safe Start)
# ---------------------------------------------------------

flask_app = Flask('')
@flask_app.route('/')
def home(): return "Bot Online"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    flask_app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    # 1. Start Web Server FIRST (Crucial for Render)
    Thread(target=run_flask).start()
    
    # 2. Try to Init Services (Database/AI) - won't crash app if fails
    print("⚙️ Initializing Services...")
    init_services()
    
    # 3. Start Bot Polling
    if TELEGRAM_BOT_TOKEN:
        print("🚀 Bot Polling Starting...")
        application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        application.add_handler(CommandHandler('start', start))
        application.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
        application.add_handler(CallbackQueryHandler(handle_callback_query))
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_menu_click))
        application.run_polling()
    else:
        print("❌ Bot Token Missing!")