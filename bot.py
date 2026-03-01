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
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

        if PINECONE_API_KEY and GOOGLE_API_KEY:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            pinecone_index = pc.Index(PINECONE_INDEX_NAME)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)
            vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings)
            logger.info("✅ Pinecone Services Initialized")
    except Exception as e:
        logger.error(f"❌ Service Init Error: {e}")

# UI Layout
MAIN_MENU_KEYBOARD = ReplyKeyboardMarkup(
    [
        [KeyboardButton("🧠 My Brain"), KeyboardButton("🤖 AI Chat")],
        [KeyboardButton("✉️ Email Draft"), KeyboardButton("📝 Summarize")],
        [KeyboardButton("🇬🇧⇄🇲🇲 Translate"), KeyboardButton("🧾 Report")]
    ],
    resize_keyboard=True
)

# Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['mode'] = None
    await update.message.reply_text("မင်္ဂလာပါ Boss! AI Assistant System အဆင်သင့်ပါ။", reply_markup=MAIN_MENU_KEYBOARD)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    mode = context.user_data.get('mode')

    # --- 1. Special Modes (Tools) ---
    if mode == 'add_link':
        if text.startswith("http"): await process_link(update, context, text)
        else: await update.message.reply_text("❌ Link အမှန် မဟုတ်ပါ (http...)", reply_markup=MAIN_MENU_KEYBOARD)
        context.user_data['mode'] = None
        return

    elif mode == 'delete_data':
        # 🗑️ DELETE LOGIC HERE
        try:
            await update.message.reply_text(f"🗑️ Deleting data for source: {text} ...")
            pinecone_index.delete(filter={"source": {"$eq": text}})
            await update.message.reply_text(f"✅ Successfully deleted data for: {text}", reply_markup=MAIN_MENU_KEYBOARD)
        except Exception as e:
            await update.message.reply_text(f"❌ Error deleting: {e}", reply_markup=MAIN_MENU_KEYBOARD)
        context.user_data['mode'] = None
        return

    # (Other Tools: Email, Summarize, etc. - Shortened for brevity, same logic as before)
    elif mode == 'email_draft':
        await call_ai(update, f"Draft a professional email about: '{text}'.")
        context.user_data['mode'] = None; return
    elif mode == 'summarize':
        await call_ai(update, f"Summarize this text: '{text}'.")
        context.user_data['mode'] = None; return
    elif mode == 'translate':
        await call_ai(update, f"Translate: '{text}'.")
        context.user_data['mode'] = None; return
    elif mode == 'report':
        await call_ai(update, f"Create a report on: '{text}'.")
        context.user_data['mode'] = None; return

    # --- 2. Main Menu Logic ---
    if text == "🧠 My Brain":
        keyboard = [
            [InlineKeyboardButton("📥 Add PDF/Word", callback_data="add_doc"), InlineKeyboardButton("🔗 Add Link", callback_data="add_link")],
            [InlineKeyboardButton("📊 Stats", callback_data="list_mem"), InlineKeyboardButton("🗑️ Delete Data", callback_data="del_data")]
        ]
        await update.message.reply_text("🧠 **My Brain Panel:**", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
        
    elif text == "🤖 AI Chat":
        context.user_data['mode'] = None
        await update.message.reply_text("🤖 မေးခွန်းမေးပါ Boss...", reply_markup=MAIN_MENU_KEYBOARD)

    # Tool Buttons
    elif text == "✉️ Email Draft": context.user_data['mode'] = 'email_draft'; await ask_input(update, "Topic for Email?")
    elif text == "📝 Summarize": context.user_data['mode'] = 'summarize'; await ask_input(update, "Text to Summarize?")
    elif text == "🇬🇧⇄🇲🇲 Translate": context.user_data['mode'] = 'translate'; await ask_input(update, "Text to Translate?")
    elif text == "🧾 Report": context.user_data['mode'] = 'report'; await ask_input(update, "Topic for Report?")

    else:
        # Default Chat (RAG)
        if not vector_store: await update.message.reply_text("⚠️ DB Error"); return
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        try:
            docs = vector_store.similarity_search(text, k=3)
            context = "\n".join([d.page_content for d in docs])
            prompt = f"Context: {context}\nQuestion: {text}\nAnswer:"
            response = llm.invoke(prompt)
            await update.message.reply_text(response.content, reply_markup=MAIN_MENU_KEYBOARD)
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

async def call_ai(update, prompt):
    response = llm.invoke(prompt)
    await update.message.reply_text(response.content, reply_markup=MAIN_MENU_KEYBOARD)

async def ask_input(update, msg):
    await update.message.reply_text(f"{msg}", reply_markup=ReplyKeyboardMarkup([[KeyboardButton("🔙 Back")]], resize_keyboard=True))

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == "del_data":
        context.user_data['mode'] = 'delete_data'
        await query.edit_message_text("🗑️ ဖျက်လိုသော File ၏ Source Path အပြည့်အစုံကို ပို့ပေးပါ Boss။\n(Example: D:\\TranscendSync\\File.docx)")
    
    # ... (Other callbacks same as before: add_doc, add_link, list_mem) ...
    elif query.data == "add_doc": await query.edit_message_text("📥 PDF/Word ပို့ပါ Boss။")
    elif query.data == "add_link": context.user_data['mode'] = 'add_link'; await query.edit_message_text("🔗 Link ပို့ပါ Boss။")
    elif query.data == "list_mem": 
        stats = pinecone_index.describe_index_stats()
        await query.edit_message_text(f"Vectors: {stats.get('total_vector_count')}")

# Process Link & Document (Same as previous code) ...
async def process_link(update, context, url): 
    # (Include previous process_link code here)
    pass 
async def handle_document(update, context):
    # (Include previous handle_document code here)
    pass

# Flask Server
flask_app = Flask('')
@flask_app.route('/')
def home(): return "Bot Running"
def run_flask(): flask_app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))

if __name__ == '__main__':
    Thread(target=run_flask).start()
    init_services()
    if TELEGRAM_BOT_TOKEN:
        app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        app.add_handler(CommandHandler('start', start))
        app.add_handler(CallbackQueryHandler(handle_callback_query))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
        # Add Document handler if needed
        app.run_polling()