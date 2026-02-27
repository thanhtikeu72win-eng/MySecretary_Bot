import os
import logging
import asyncio
import tempfile
from telegram import Update, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from flask import Flask
from threading import Thread

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

# Debug Check
print(f"DEBUG CHECK: TELEGRAM_BOT_TOKEN is {'✅ OK' if TELEGRAM_BOT_TOKEN else '❌ MISSING'}")
print(f"DEBUG CHECK: GOOGLE_API_KEY is {'✅ OK' if GOOGLE_API_KEY else '❌ MISSING'}")
print(f"DEBUG CHECK: SUPABASE_URL is {'✅ OK' if SUPABASE_URL else '❌ MISSING'}")
print(f"DEBUG CHECK: SUPABASE_KEY is {'✅ OK' if SUPABASE_KEY else '❌ MISSING'}")

if not all([TELEGRAM_BOT_TOKEN, GOOGLE_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    print("❌ Error: Missing Environment Variables!")

# 3. Initialize Clients
genai.configure(api_key=GOOGLE_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- CRITICAL FIX: Using the correct, newer embedding model ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

# ---------------------------------------------------------
# UI Layout (Persistent Menu)
# ---------------------------------------------------------

# This keyboard will be attached to EVERY message
MAIN_MENU_KEYBOARD = ReplyKeyboardMarkup(
    [
        ["🧠 My Brain", "🤖 AI Assistant"],
        ["📅 My Schedule", "⚡ Utilities"]
    ],
    resize_keyboard=True,
    one_time_keyboard=False # This keeps the menu visible!
)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

async def process_document(update: Update, context: ContextTypes.DEFAULT_TYPE, texts, source_name):
    """Saves text to Supabase (My Brain)"""
    try:
        status_msg = await update.message.reply_text(f"⏳ Saving {source_name} to Brain...", reply_markup=MAIN_MENU_KEYBOARD)
        vector_store.add_documents(texts)
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=status_msg.message_id, text=f"✅ Saved: {source_name}")
    except Exception as e:
        logging.error(f"Error saving document: {e}")
        await update.message.reply_text(f"❌ Brain Error: {str(e)}", reply_markup=MAIN_MENU_KEYBOARD)

# ---------------------------------------------------------
# Handlers
# ---------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['mode'] = None
    welcome_msg = "မင်္ဂလာပါ Boss! 'My Brain' စနစ် အဆင်သင့်ဖြစ်ပါပြီ။"
    await update.message.reply_text(welcome_msg, reply_markup=MAIN_MENU_KEYBOARD)

async def handle_menu_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles Main Menu Clicks"""
    text = update.message.text
    context.user_data['mode'] = None # Reset any previous mode

    # --- FOCUS AREA: MY BRAIN ---
    if text == "🧠 My Brain":
        keyboard = [
            [InlineKeyboardButton("📥 Add PDF", callback_data="add_pdf"), InlineKeyboardButton("🔗 Add Link", callback_data="add_link")],
            [InlineKeyboardButton("🧹 Clear Memory", callback_data="clear_memory")]
        ]
        await update.message.reply_text(
            "🧠 **My Brain Control Panel:**\n\n"
            "• **Add PDF:** စာရွက်စာတမ်းတွေ မှတ်ခိုင်းမယ်။\n"
            "• **Add Link:** Website တွေကို ဖတ်ခိုင်းမယ်။\n"
            "• **Clear:** မှတ်ထားသမျှ ဖျက်မယ်။",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

    # --- Placeholders for now ---
    elif text == "🤖 AI Assistant":
        await update.message.reply_text("🤖 AI Assistant is Next Step.", reply_markup=MAIN_MENU_KEYBOARD)
    elif text == "📅 My Schedule":
        await update.message.reply_text("📅 Schedule is Future Step.", reply_markup=MAIN_MENU_KEYBOARD)
    elif text == "⚡ Utilities":
        await update.message.reply_text("⚡ Utilities is Future Step.", reply_markup=MAIN_MENU_KEYBOARD)
    
    # --- Chat Logic (Uses Brain if available) ---
    else:
        # If user types normal text, check Brain first (RAG)
        await update.message.reply_chat_action("typing")
        try:
            # Check for Link input
            if context.user_data.get('expecting') == 'link':
                # Process Link Logic Here (Simplified for brevity, usually handled in separate logic)
                context.user_data['expecting'] = None
                await update.message.reply_text("🔗 Link functionality coming in next update.", reply_markup=MAIN_MENU_KEYBOARD)
                return

            # Normal RAG Chat
            related_docs = vector_store.similarity_search(text, k=3)
            context_text = "\n\n".join([doc.page_content for doc in related_docs])
            
            if context_text:
                prompt = f"Answer based on this context:\n{context_text}\n\nQuestion: {text}"
                await update.message.reply_text("🧠 (Using Brain Memory)...", reply_markup=MAIN_MENU_KEYBOARD)
            else:
                prompt = text # General Chat
            
            response = llm.invoke(prompt)
            await update.message.reply_text(response.content, reply_markup=MAIN_MENU_KEYBOARD)

        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}", reply_markup=MAIN_MENU_KEYBOARD)

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    
    # --- MY BRAIN ACTIONS ---
    if data == "add_pdf":
        await query.edit_message_text("📥 PDF ဖိုင်ကို အခု ပို့ပေးပါ Boss။")
        # Note: The 'handle_pdf' function below will catch the file automatically.
    
    elif data == "add_link":
        context.user_data['expecting'] = 'link'
        await query.edit_message_text("🔗 Link ကို Copy ကူးပြီး ချပေးပါ Boss။")

    elif data == "clear_memory":
        # Warning: This needs Supabase delete logic, keeping it simple for now
        await query.edit_message_text("🧹 Memory Clean function implementation needed (Database Reset).")
    
    else:
        await query.edit_message_text("This feature is for the next step.")

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles PDF Uploads for My Brain"""
    document = update.message.document
    if document.mime_type != 'application/pdf':
        await update.message.reply_text("PDF Only please!", reply_markup=MAIN_MENU_KEYBOARD)
        return

    msg = await update.message.reply_text("📥 Processing PDF...", reply_markup=MAIN_MENU_KEYBOARD)
    
    try:
        file = await context.bot.get_file(document.file_id)
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
            await file.download_to_drive(custom_path=temp_pdf.name)
            loader = PyPDFLoader(temp_pdf.name)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(pages)
            for doc in texts: doc.metadata = {"source": document.file_name}
            
            # Save to Supabase
            vector_store.add_documents(texts)
            
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text=f"✅ Success! I have read '{document.file_name}'.")
    except Exception as e:
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text=f"❌ Error: {str(e)}")

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

flask_app = Flask('')
@flask_app.route('/')
def home(): return "Secretary Bot (Brain Mode) is Online"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    flask_app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    Thread(target=run_flask).start()
    
    print("🚀 Bot starting...")
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
    application.add_handler(CallbackQueryHandler(handle_callback_query))
    # Handles Text AND keeps menu persistent
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_menu_click))
    
    application.run_polling()