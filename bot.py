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
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client

# 1. Setup Logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Load Env Vars
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Debug Check
print(f"DEBUG CHECK: TELEGRAM_BOT_TOKEN is {'✅ OK' if TELEGRAM_BOT_TOKEN else '❌ MISSING'}")
print(f"DEBUG CHECK: GOOGLE_API_KEY is {'✅ OK' if GOOGLE_API_KEY else '❌ MISSING'}")
print(f"DEBUG CHECK: SUPABASE_URL is {'✅ OK' if SUPABASE_URL else '❌ MISSING'}")
print(f"DEBUG CHECK: SUPABASE_KEY is {'✅ OK' if SUPABASE_KEY else '❌ MISSING'}")

# 3. Global Vars
vector_store = None
llm = None
supabase = None # Client for raw queries

def init_services():
    global vector_store, llm, supabase
    try:
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

        if SUPABASE_URL and SUPABASE_KEY:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
            vector_store = SupabaseVectorStore(client=supabase, embedding=embeddings, table_name="documents", query_name="match_documents")
            logger.info("✅ Services Initialized")
    except Exception as e:
        logger.error(f"❌ Service Init Error: {e}")

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
    await update.message.reply_text("မင်္ဂလာပါ Boss! စနစ် အဆင်သင့်ပါ။", reply_markup=MAIN_MENU_KEYBOARD)

async def handle_menu_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    
    if text == "🧠 My Brain":
        keyboard = [
            [InlineKeyboardButton("📥 Add PDF/Word", callback_data="add_doc"), InlineKeyboardButton("🗂 List Memory", callback_data="list_mem")],
            [InlineKeyboardButton("🧹 Clear All", callback_data="clear_mem")]
        ]
        await update.message.reply_text("🧠 **My Brain Panel:**", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
        
    elif text == "🤖 AI Assistant":
        await update.message.reply_text("🤖 AI Feature coming next.")
    
    else:
        # Chat Logic (RAG)
        if not vector_store: return
        await update.message.reply_chat_action("typing")
        try:
            docs = vector_store.similarity_search(text, k=2)
            context_text = "\n".join([d.page_content for d in docs])
            prompt = f"Context: {context_text}\n\nQuestion: {text}" if context_text else text
            response = llm.invoke(prompt)
            await update.message.reply_text(response.content, reply_markup=MAIN_MENU_KEYBOARD)
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == "add_doc":
        await query.edit_message_text("📥 PDF သို့မဟုတ် Word ဖိုင် ပို့ပေးပါ Boss။")

    elif query.data == "list_mem":
        # Query Supabase for unique sources
        if not supabase:
            await query.edit_message_text("❌ Database Connection Error.")
            return
            
        try:
            # Fetch metadata from documents table (Limit to last 100 entries to avoid overflow)
            response = supabase.table("documents").select("metadata").limit(100).execute()
            data = response.data
            
            # Extract unique sources
            sources = set()
            for row in data:
                meta = row.get('metadata', {})
                if 'source' in meta:
                    sources.add(meta['source'])
            
            if not sources:
                await query.edit_message_text("📭 My Brain မှာ ဘာမှ မရှိသေးပါဘူး Boss။")
            else:
                text_list = "\n".join([f"• {s}" for s in sources])
                await query.edit_message_text(f"🗂 **မှတ်သားထားသော ဖိုင်များ:**\n\n{text_list}", parse_mode="Markdown")
                
        except Exception as e:
            await query.edit_message_text(f"❌ List Error: {str(e)}")

    elif query.data == "clear_mem":
        await query.edit_message_text("🧹 Database ကို ဖျက်ဖို့အတွက် Supabase Dashboard ကနေ လုပ်တာ ပိုစိတ်ချရပါတယ် Boss။")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not vector_store: return
    document = update.message.document
    file_name = document.file_name
    
    # Check file type
    if not (file_name.endswith('.pdf') or file_name.endswith('.docx')):
        await update.message.reply_text("❌ PDF သို့မဟုတ် DOCX ဖိုင်ပဲ လက်ခံပါတယ် Boss။")
        return

    status = await update.message.reply_text(f"📥 Reading {file_name}...", reply_markup=MAIN_MENU_KEYBOARD)
    
    try:
        file = await context.bot.get_file(document.file_id)
        
        # Determine loader based on extension
        with tempfile.NamedTemporaryFile(delete=True, suffix=os.path.splitext(file_name)[1]) as temp_file:
            await file.download_to_drive(custom_path=temp_file.name)
            
            if file_name.endswith('.pdf'):
                loader = PyPDFLoader(temp_file.name)
            else:
                loader = Docx2txtLoader(temp_file.name) # Requires docx2txt
                
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(pages)
            
            # Add metadata
            for doc in texts: 
                doc.metadata = {"source": file_name}
            
            # Save to DB
            vector_store.add_documents(texts)
            
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status.message_id, text=f"✅ Saved: {file_name}")
        
    except Exception as e:
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status.message_id, text=f"❌ Error: {str(e)}")

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
flask_app = Flask('')
@flask_app.route('/')
def home(): return "Bot Online"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    flask_app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    Thread(target=run_flask).start()
    init_services()
    if TELEGRAM_BOT_TOKEN:
        application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        application.add_handler(CommandHandler('start', start))
        # Handle both PDF and DOCX via generic Document filter
        application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
        application.add_handler(CallbackQueryHandler(handle_callback_query))
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_menu_click))
        application.run_polling()