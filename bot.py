import os
import logging
import asyncio
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
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legal-bot") # Default name if not set

# Debug Check
print(f"DEBUG CHECK: TELEGRAM_BOT_TOKEN is {'✅ OK' if TELEGRAM_BOT_TOKEN else '❌ MISSING'}")
print(f"DEBUG CHECK: GOOGLE_API_KEY is {'✅ OK' if GOOGLE_API_KEY else '❌ MISSING'}")
print(f"DEBUG CHECK: PINECONE_API_KEY is {'✅ OK' if PINECONE_API_KEY else '❌ MISSING'}")

# 3. Global Vars
vector_store = None
llm = None
pinecone_index = None

def init_services():
    global vector_store, llm, pinecone_index
    try:
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
            # Note: gemini-2.5-flash is not widely public yet, using 1.5-flash or pro for safety. 
            # If you have access to 1.5-pro or 2.0, change it back.

        if PINECONE_API_KEY and GOOGLE_API_KEY:
            # Initialize Pinecone Client
            pc = Pinecone(api_key=PINECONE_API_KEY)
            pinecone_index = pc.Index(PINECONE_INDEX_NAME)
            
            # Initialize Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)
            
            # Initialize Vector Store
            vector_store = PineconeVectorStore(
                index=pinecone_index,
                embedding=embeddings
            )
            logger.info("✅ Pinecone Services Initialized")
            
    except Exception as e:
        logger.error(f"❌ Service Init Error: {e}")

# ---------------------------------------------------------
# UI Layout (Persistent Keyboard)
# ---------------------------------------------------------
MAIN_MENU_KEYBOARD = ReplyKeyboardMarkup(
    [
        [KeyboardButton("🧠 My Brain"), KeyboardButton("🤖 AI Assistant")],
        [KeyboardButton("📅 My Schedule"), KeyboardButton("⚡ Utilities")]
    ],
    resize_keyboard=True,
    one_time_keyboard=False # အရေးကြီး: ခလုတ်မပျောက်အောင်
)

# ---------------------------------------------------------
# Handlers
# ---------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['mode'] = None
    await update.message.reply_text(
        "မင်္ဂလာပါ Boss! Pinecone System အဆင်သင့်ပါ။", 
        reply_markup=MAIN_MENU_KEYBOARD
    )

async def handle_menu_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    
    # Check if user is in 'Adding Link' mode
    if context.user_data.get('mode') == 'add_link':
        if text.startswith("http"):
            await process_link(update, context, text)
        else:
            await update.message.reply_text(
                "❌ Link အမှန် မဟုတ်ပါဘူး Boss။ (http နဲ့ စရပါမယ်)", 
                reply_markup=MAIN_MENU_KEYBOARD
            )
        context.user_data['mode'] = None # Reset mode
        return

    # Normal Menu Logic
    if text == "🧠 My Brain":
        keyboard = [
            [InlineKeyboardButton("📥 Add PDF/Word", callback_data="add_doc"), InlineKeyboardButton("🔗 Add Link", callback_data="add_link")],
            [InlineKeyboardButton("📊 Memory Stats", callback_data="list_mem"), InlineKeyboardButton("🧹 Help", callback_data="help_mem")]
        ]
        await update.message.reply_text(
            "🧠 **My Brain Panel:**\nစာရွက်စာတမ်းသစ်များ ထည့်သွင်းနိုင်ပါသည်။", 
            reply_markup=InlineKeyboardMarkup(keyboard), 
            parse_mode="Markdown"
        )
        
    elif text == "🤖 AI Assistant":
        await update.message.reply_text("🤖 မေးခွန်းမေးပါ Boss...", reply_markup=MAIN_MENU_KEYBOARD)
    
    elif text == "📅 My Schedule":
        await update.message.reply_text("📅 Schedule feature coming soon.", reply_markup=MAIN_MENU_KEYBOARD)

    elif text == "⚡ Utilities":
         await update.message.reply_text("⚡ Tools feature coming soon.", reply_markup=MAIN_MENU_KEYBOARD)
    
    else:
        # Chat Logic (RAG)
        if not vector_store: 
            await update.message.reply_text("⚠️ Database မချိတ်ရသေးပါ Boss။", reply_markup=MAIN_MENU_KEYBOARD)
            return

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        try:
            # 1. Search in Pinecone
            docs = vector_store.similarity_search(text, k=3)
            context_text = "\n\n".join([d.page_content for d in docs])
            
            # 2. Generate Answer
            if context_text:
                prompt = f"ဖြေဆိုရန် အချက်အလက်များ:\n{context_text}\n\nမေးခွန်း: {text}\n\n(အထက်ပါ အချက်အလက်များကို အခြေခံ၍ မြန်မာလို ဖြေပေးပါ)"
            else:
                prompt = text # No context found, just ask AI directly

            response = llm.invoke(prompt)
            
            # 3. Reply with Permanent Keyboard
            await update.message.reply_text(response.content, reply_markup=MAIN_MENU_KEYBOARD)
            
        except Exception as e:
            logger.error(f"AI Error: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}", reply_markup=MAIN_MENU_KEYBOARD)

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == "add_doc":
        await query.edit_message_text("📥 PDF သို့မဟုတ် Word ဖိုင် ပို့ပေးပါ Boss။")

    elif query.data == "add_link":
        context.user_data['mode'] = 'add_link'
        await query.edit_message_text("🔗 Web Link (URL) ကို Copy ကူးပြီး ပို့ပေးပါ Boss။")

    elif query.data == "list_mem":
        # Pinecone can't list files easily like SQL, so we show Stats
        try:
            stats = pinecone_index.describe_index_stats()
            count = stats.get('total_vector_count', 0)
            await query.edit_message_text(f"📊 **Memory Status:**\n\nTotal Vectors: {count}\n(Pinecone doesn't support listing filenames directly)", parse_mode="Markdown")
        except Exception as e:
            await query.edit_message_text(f"❌ Stats Error: {str(e)}")

    elif query.data == "help_mem":
        await query.edit_message_text("🧹 Data တွေကို Pinecone Dashboard ကနေ ဝင်ဖျက်လို့ ရပါတယ် Boss။")

async def process_link(update: Update, context: ContextTypes.DEFAULT_TYPE, url):
    """Handles Web Link Processing"""
    if not vector_store: return
    status = await update.message.reply_text(f"🔗 Reading Website: {url}...", reply_markup=MAIN_MENU_KEYBOARD)
    
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(docs)
        for doc in texts: doc.metadata = {"source": url}
        
        vector_store.add_documents(texts)
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status.message_id, text=f"✅ Saved Link: {url}")
        # Send follow up to restore keyboard cleanly if needed
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Ready next!", reply_markup=MAIN_MENU_KEYBOARD)

    except Exception as e:
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status.message_id, text=f"❌ Link Error: {str(e)}")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not vector_store: 
        await update.message.reply_text("⚠️ Database connection missing.", reply_markup=MAIN_MENU_KEYBOARD)
        return

    document = update.message.document
    file_name = document.file_name
    
    if not (file_name.lower().endswith('.pdf') or file_name.lower().endswith('.docx')):
        await update.message.reply_text("❌ PDF/DOCX Only!", reply_markup=MAIN_MENU_KEYBOARD)
        return

    status = await update.message.reply_text(f"📥 Reading {file_name}...", reply_markup=MAIN_MENU_KEYBOARD)
    try:
        file = await context.bot.get_file(document.file_id)
        # Use tempfile to handle file safely
        with tempfile.NamedTemporaryFile(delete=True, suffix=os.path.splitext(file_name)[1]) as temp_file:
            await file.download_to_drive(custom_path=temp_file.name)
            
            if file_name.lower().endswith('.pdf'): 
                loader = PyPDFLoader(temp_file.name)
            else: 
                loader = Docx2txtLoader(temp_file.name)
            
            pages = loader.load()
            if not pages:
                 await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status.message_id, text=f"⚠️ Empty File: {file_name}")
                 return

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(pages)
            for doc in texts: doc.metadata = {"source": file_name}
            
            vector_store.add_documents(texts)
            
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status.message_id, text=f"✅ Saved: {file_name}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="နောက်ထပ် ဘာလုပ်ပေးရမလဲ Boss?", reply_markup=MAIN_MENU_KEYBOARD)

    except Exception as e:
        logger.error(f"Doc Error: {e}")
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status.message_id, text=f"❌ Error: {str(e)}")

# ---------------------------------------------------------
# Flask Keep-Alive Server
# ---------------------------------------------------------
flask_app = Flask('')

@flask_app.route('/')
def home():
    return "Bot is Alive & Running!"

def run_flask():
    # Render provides PORT env var, defaults to 10000 if not set
    port = int(os.environ.get("PORT", 10000))
    flask_app.run(host='0.0.0.0', port=port)

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
if __name__ == '__main__':
    # 1. Start Flask in Background
    Thread(target=run_flask).start()
    
    # 2. Init AI Services
    init_services()
    
    # 3. Start Bot
    if TELEGRAM_BOT_TOKEN:
        print("🚀 Starting Telegram Bot...")
        application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Handlers
        application.add_handler(CommandHandler('start', start))
        application.add_handler(CallbackQueryHandler(handle_callback_query))
        
        # Documents (PDF/DOCX)
        application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
        
        # Text Messages (Menu & Chat)
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_menu_click))
        
        application.run_polling()
    else:
        print("❌ Error: TELEGRAM_BOT_TOKEN is missing!")