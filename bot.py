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
    raise ValueError("Missing environment variables! Please check Render Settings.")

# 3. Initialize Clients
genai.configure(api_key=GOOGLE_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Setup Embeddings & Vector Store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

# Setup Chat Model with "Secretary Persona"
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
# UI Layout Definition (Persistent Main Menu)
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
    """Common function to save texts to Supabase"""
    try:
        await update.message.reply_text(f"⏳ {source_name} ကို မှတ်တမ်းတင်နေပါပြီ Boss... ခဏစောင့်ပေးနော်။", reply_markup=MAIN_MENU_KEYBOARD)
        vector_store.add_documents(texts)
        await update.message.reply_text(f"✅ {source_name} ကို ဦးနှောက်ထဲ ထည့်ပြီးပါပြီရှင်။", reply_markup=MAIN_MENU_KEYBOARD)
    except Exception as e:
        logging.error(f"Error saving document: {e}")
        await update.message.reply_text(f"❌ မှတ်သားရာမှာ အမှားရှိနေပါတယ်ရှင်: {str(e)}", reply_markup=MAIN_MENU_KEYBOARD)

# ---------------------------------------------------------
# Handlers
# ---------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Shows the Main Menu with Persona"""
    # Clear any previous modes
    context.user_data['mode'] = None
    
    welcome_msg = (
        "မင်္ဂလာပါ Boss! 🙏\n"
        "ကျွန်မက Boss ရဲ့ ကိုယ်ပိုင် အတွင်းရေးမှူးပါရှင်။ \n\n"
        "Boss ရဲ့ စာရွက်စာတမ်းတွေ သိမ်းပေးတာ၊ အီးမေးလ် ရေးပေးတာနဲ့ "
        "တခြား ကိစ္စတွေအတွက် အောက်က Menu ကနေ ခိုင်းစေနိုင်ပါတယ်ရှင်။"
    )
    await update.message.reply_text(welcome_msg, reply_markup=MAIN_MENU_KEYBOARD)

async def handle_menu_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles clicks on the Main Menu buttons"""
    text = update.message.text
    
    # Always reset mode when clicking main menu
    context.user_data['mode'] = None

    if text == "🧠 My Brain":
        keyboard = [
            [InlineKeyboardButton("📥 Add PDF", callback_data="add_pdf"), InlineKeyboardButton("🔗 Add Link", callback_data="add_link")],
            [InlineKeyboardButton("🗂 List Knowledge", callback_data="list_knowledge"), InlineKeyboardButton("🧹 Clear All", callback_data="clear_knowledge")]
        ]
        await update.message.reply_text("🧠 **Knowledge Management:**\nစာရွက်စာတမ်းတွေနဲ့ Link တွေကို ဘယ်လို စီမံပေးရမလဲရှင်?", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")

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
        await update.message.reply_text("📅 **Schedule Manager:**\nရက်ချိန်းတွေ မှတ်ထားမလား Boss?", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")

    elif text == "⚡ Utilities":
        keyboard = [
            [InlineKeyboardButton("🌤 Weather", callback_data="weather"), InlineKeyboardButton("💱 Currency", callback_data="currency")],
            [InlineKeyboardButton("📞 Call Boss", url="tel:+95912345678")]
        ]
        await update.message.reply_text("⚡ **Utilities:**\nရာသီဥတုနဲ့ ငွေဈေးနှုန်းတွေ ကြည့်မလားရှင်?", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")

    else:
        # Not a button click? Treat as Chat/RAG
        await handle_rag_chat(update, context)

async def handle_rag_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles normal text messages (Chat / RAG / AI Tools)"""
    user_text = update.message.text
    
    # Check current mode (AI Tools)
    mode = context.user_data.get('mode')
    
    if mode == "draft_email":
        await update.message.reply_chat_action("typing")
        prompt = f"Act as a professional secretary. Draft a formal email about: {user_text}. Make it polished and clear."
        response = llm.invoke(prompt)
        await update.message.reply_text(f"✉️ **Email Draft:**\n\n{response.content}", parse_mode="Markdown", reply_markup=MAIN_MENU_KEYBOARD)
        context.user_data['mode'] = None # Reset mode
        return

    elif mode == "translate":
        await update.message.reply_chat_action("typing")
        prompt = f"Translate the following text. If it is Burmese, translate to English. If it is English, translate to Burmese:\n\n{user_text}"
        response = llm.invoke(prompt)
        await update.message.reply_text(f"🔤 **Translation:**\n\n{response.content}", reply_markup=MAIN_MENU_KEYBOARD)
        context.user_data['mode'] = None
        return

    elif mode == "summarize":
        await update.message.reply_chat_action("typing")
        prompt = f"Summarize the following text into key bullet points (in Burmese):\n\n{user_text}"
        response = llm.invoke(prompt)
        await update.message.reply_text(f"📝 **Summary:**\n\n{response.content}", reply_markup=MAIN_MENU_KEYBOARD)
        context.user_data['mode'] = None
        return
        
    elif mode == "report":
        await update.message.reply_chat_action("typing")
        prompt = f"Create a professional report outline for the topic: {user_text}. Use clear headings."
        response = llm.invoke(prompt)
        await update.message.reply_text(f"🧾 **Report Outline:**\n\n{response.content}", parse_mode="Markdown", reply_markup=MAIN_MENU_KEYBOARD)
        context.user_data['mode'] = None
        return

    # Normal Chat / RAG Logic (if no mode is set)
    
    # 1. Check for URL
    if user_text.startswith("http"):
        await update.message.reply_text(f"🔗 Link ကို ဖတ်နေပါတယ်ရှင်...: {user_text}", reply_markup=MAIN_MENU_KEYBOARD)
        try:
            loader = WebBaseLoader(user_text)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(docs)
            for doc in texts: doc.metadata = {"source": user_text}
            await process_document(update, context, texts, "Website")
        except Exception as e:
            await update.message.reply_text(f"❌ Link ဖတ်မရလို့ပါရှင်: {str(e)}", reply_markup=MAIN_MENU_KEYBOARD)
        return

    # 2. RAG Search & Chat
    await update.message.reply_chat_action("typing")
    try:
        related_docs = vector_store.similarity_search(user_text, k=3)
        context_text = "\n\n".join([doc.page_content for doc in related_docs])
        
        # Inject Persona into Prompt
        if not context_text:
            prompt = f"{secretary_instruction}\n\nUser Question: {user_text}"
        else:
            prompt = f"""{secretary_instruction}
            
            Based ONLY on the following context, answer the Boss's question:
            
            Context: {context_text}
            
            Boss's Question: {user_text}
            """
        
        response = llm.invoke(prompt)
        await update.message.reply_text(response.content, reply_markup=MAIN_MENU_KEYBOARD)

    except Exception as e:
        logging.error(f"Error: {e}")
        await update.message.reply_text("❌ စနစ်ပိုင်းဆိုင်ရာ အနည်းငယ် Error ရှိနေပါတယ်ရှင်။", reply_markup=MAIN_MENU_KEYBOARD)

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles clicks on Inline Keyboard buttons (Sub-menus)"""
    query = update.callback_query
    await query.answer() 
    data = query.data
    
    # AI Tools Logic (Setting Modes)
    if data == "draft_email":
        context.user_data['mode'] = "draft_email"
        await query.edit_message_text("✉️ Email ရေးဖို့ အကြောင်းအရာ (Topic) ကို ရိုက်ထည့်ပေးပါရှင်။\n(ဥပမာ: Sick leave request to manager)")
    
    elif data == "translate":
        context.user_data['mode'] = "translate"
        await query.edit_message_text("🇬🇧⇄🇲🇲 ဘာသာပြန်ပေးပါမယ်။ စာသားကို ရိုက်ထည့်ပေးပါရှင်။")

    elif data == "summarize":
        context.user_data['mode'] = "summarize"
        await query.edit_message_text("📝 အနှစ်ချုပ်ပေးရမယ့် စာသား (သို့) Link ကို ပေးပါရှင်။")

    elif data == "report":
        context.user_data['mode'] = "report"
        await query.edit_message_text("🧾 Report ရေးဖို့ ခေါင်းစဉ် (Topic) ပြောပေးပါရှင်။")

    # Knowledge Logic (Placeholder)
    elif data == "add_pdf":
        await query.edit_message_text("📥 PDF ဖိုင်ကို ပို့ပေးပါရှင်။ ကျွန်မ သိမ်းထားလိုက်ပါ့မယ်။")
    
    # Utilities (Placeholder)
    elif data == "weather":
        await query.edit_message_text("🌤 Weather feature coming soon! (Need API Key)")
    
    else:
        await query.edit_message_text(f"Boss ရွေးလိုက်တဲ့ '{data}' ကို ပြင်ဆင်နေဆဲပါရှင် 🚧")

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle PDF file uploads"""
    document = update.message.document
    if document.mime_type != 'application/pdf':
        await update.message.reply_text("PDF ဖိုင်ပဲ လက်ခံပါတယ်ရှင်။", reply_markup=MAIN_MENU_KEYBOARD)
        return

    await update.message.reply_text("📥 PDF ကို လက်ခံရရှိပါတယ်ရှင်...", reply_markup=MAIN_MENU_KEYBOARD)
    file = await context.bot.get_file(document.file_id)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
        await file.download_to_drive(custom_path=temp_pdf.name)
        loader = PyPDFLoader(temp_pdf.name)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(pages)
        for doc in texts: doc.metadata = {"source": document.file_name}
        await process_document(update, context, texts, document.file_name)

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

if __name__ == '__main__':
    from flask import Flask
    from threading import Thread

    flask_app = Flask('')
    @flask_app.route('/')
    def home(): return "Secretary Bot is online!"
    
    def run_flask():
        port = int(os.environ.get("PORT", 10000))
        flask_app.run(host='0.0.0.0', port=port)

    Thread(target=run_flask).start()

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
    application.add_handler(CallbackQueryHandler(handle_callback_query))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_menu_click))

    print("Secretary Bot is ready to serve...")
    application.run_polling()