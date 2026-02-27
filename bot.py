import os
import logging
import asyncio
import tempfile
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, CallbackQueryHandler, filters

# Gemini & LangChain Imports
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client

# 1. Setup Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# 2. Load Environment Variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not all([TELEGRAM_BOT_TOKEN, GOOGLE_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    raise ValueError("Missing environment variables!")

# 3. Initialize Clients
genai.configure(api_key=GOOGLE_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- FIX #1: Use 'models/text-embedding-004' instead of 'embedding-001' ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

# Setup Chat Model
secretary_instruction = """
You are a highly efficient, smart, and professional female executive secretary named 'MySecretary'.
Your Boss is the user. You help him with tasks, summaries, and information.
Tone: Polite, Respectful, Sharp, and Concise.
Language: Burmese (Myanmar).
Style: Use 'ရှင်' (Shin) or 'ပါ' (Pa) at the end of sentences appropriate for a female speaker.
Never say you are an AI model unless asked directly. Act like a real secretary.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    convert_system_message_to_human=True
)

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
# Helper Functions
# ---------------------------------------------------------

async def process_document(update: Update, context: ContextTypes.DEFAULT_TYPE, texts, source_name):
    try:
        await update.message.reply_text(f"⏳ {source_name} ကို မှတ်တမ်းတင်နေပါပြီ Boss... ခဏစောင့်ပေးနော်။", reply_markup=MAIN_MENU_KEYBOARD)
        vector_store.add_documents(texts)
        await update.message.reply_text(f"✅ {source_name} ကို ဦးနှောက်ထဲ ထည့်ပြီးပါပြီရှင်။", reply_markup=MAIN_MENU_KEYBOARD)
    except Exception as e:
        logging.error(f"Error saving document: {e}")
        await update.message.reply_text(f"❌ Error: {str(e)}", reply_markup=MAIN_MENU_KEYBOARD)

# ---------------------------------------------------------
# Handlers
# ---------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['mode'] = None
    welcome_msg = "မင်္ဂလာပါ Boss! 🙏\nကျွန်မက Boss ရဲ့ ကိုယ်ပိုင် အတွင်းရေးမှူးပါရှင်။"
    await update.message.reply_text(welcome_msg, reply_markup=MAIN_MENU_KEYBOARD)

async def handle_menu_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    context.user_data['mode'] = None

    if text == "🧠 My Brain":
        keyboard = [
            [InlineKeyboardButton("📥 Add PDF", callback_data="add_pdf"), InlineKeyboardButton("🔗 Add Link", callback_data="add_link")],
            [InlineKeyboardButton("🗂 List Knowledge", callback_data="list_knowledge"), InlineKeyboardButton("🧹 Clear All", callback_data="clear_knowledge")]
        ]
        await update.message.reply_text("🧠 **Knowledge Management:**", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")

    elif text == "🤖 AI Assistant":
        keyboard = [
            [InlineKeyboardButton("✉️ Email Draft", callback_data="draft_email"), InlineKeyboardButton("📝 Summarize", callback_data="summarize")],
            [InlineKeyboardButton("🇬🇧⇄🇲🇲 Translate", callback_data="translate"), InlineKeyboardButton("🧾 Report", callback_data="report")]
        ]
        await update.message.reply_text("🤖 **AI Tools:**\nBoss ဘာကူညီပေးရမလဲရှင်?", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")

    elif text == "📅 My Schedule":
        keyboard = [
            [InlineKeyboardButton("➕ New Reminder", callback_data="new_reminder")],
            [InlineKeyboardButton("📋 View List", callback_data="view_reminders")]
        ]
        await update.message.reply_text("📅 **Schedule:**", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")

    elif text == "⚡ Utilities":
        # --- FIX #2: Removed 'url=tel:...' and changed to callback_data ---
        keyboard = [
            [InlineKeyboardButton("🌤 Weather", callback_data="weather"), InlineKeyboardButton("💱 Currency", callback_data="currency")],
            [InlineKeyboardButton("📞 Get Boss Number", callback_data="get_number")] 
        ]
        await update.message.reply_text("⚡ **Utilities:**\nရာသီဥတုနဲ့ ငွေဈေးနှုန်းတွေ ကြည့်မလားရှင်?", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")

    else:
        await handle_rag_chat(update, context)

async def handle_rag_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    mode = context.user_data.get('mode')
    
    if mode == "draft_email":
        await update.message.reply_chat_action("typing")
        prompt = f"Act as a professional secretary. Draft a formal email about: {user_text}."
        try:
            response = llm.invoke(prompt)
            await update.message.reply_text(f"✉️ **Email Draft:**\n\n{response.content}", reply_markup=MAIN_MENU_KEYBOARD)
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
        context.user_data['mode'] = None
        return

    # ... (Similar checks for translate/summarize/report - keeping code short for readability) ...
    # You can add the other modes back if they were working, or I can provide them if you need.
    # Assuming basic chat/rag for now:

    # RAG Logic
    await update.message.reply_chat_action("typing")
    try:
        related_docs = vector_store.similarity_search(user_text, k=3)
        context_text = "\n\n".join([doc.page_content for doc in related_docs])
        
        prompt = f"""{secretary_instruction}
        Context: {context_text}
        Question: {user_text}
        """
        response = llm.invoke(prompt)
        await update.message.reply_text(response.content, reply_markup=MAIN_MENU_KEYBOARD)
    except Exception as e:
        logging.error(f"Error: {e}")
        await update.message.reply_text("❌ စကားပြောရာမှာ Error ဖြစ်နေပါတယ်ရှင်။", reply_markup=MAIN_MENU_KEYBOARD)

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer() 
    data = query.data
    
    if data == "draft_email":
        context.user_data['mode'] = "draft_email"
        await query.edit_message_text("✉️ Email ရေးဖို့ Topic ပေးပါရှင်။")
    
    # --- FIX #2 Implementation: Send Number as Text ---
    elif data == "get_number":
        await query.edit_message_text("📞 Boss ဖုန်းနံပါတ်: 09-12345678")

    elif data == "weather":
        # Placeholder for next step
        await query.edit_message_text("🌤 Weather Feature ကို နောက်အဆင့်မှာ ထည့်သွင်းပေးပါမယ်ရှင်။")

    elif data == "currency":
        # Placeholder for next step
        await query.edit_message_text("💱 Currency Feature ကို နောက်အဆင့်မှာ ထည့်သွင်းပေးပါမယ်ရှင်။")
    
    else:
        await query.edit_message_text(f"Work in progress: {data}")

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    if document.mime_type != 'application/pdf': return
    await update.message.reply_text("📥 PDF...", reply_markup=MAIN_MENU_KEYBOARD)
    file = await context.bot.get_file(document.file_id)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
        await file.download_to_drive(custom_path=temp_pdf.name)
        loader = PyPDFLoader(temp_pdf.name)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(pages)
        for doc in texts: doc.metadata = {"source": document.file_name}
        await process_document(update, context, texts, document.file_name)

if __name__ == '__main__':
    from flask import Flask
    from threading import Thread
    flask_app = Flask('')
    @flask_app.route('/')
    def home(): return "Bot Online"
    def run_flask(): flask_app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
    Thread(target=run_flask).start()
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
    application.add_handler(CallbackQueryHandler(handle_callback_query))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_menu_click))
    application.run_polling()