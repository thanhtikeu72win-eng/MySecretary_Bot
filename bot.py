import os
import logging
import asyncio
import tempfile
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Check if keys are present
if not all([TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
    raise ValueError("Missing environment variables! Check Render settings.")

# 3. Initialize Clients
genai.configure(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Setup Embeddings & Vector Store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

# Setup Chat Model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

async def process_document(update: Update, context: ContextTypes.DEFAULT_TYPE, texts, source_name):
    """Common function to save texts to Supabase"""
    try:
        await update.message.reply_text(f"⏳ Saving {len(texts)} chunks from {source_name}...")
        
        # Async add to vector store
        vector_store.add_documents(texts)
        
        await update.message.reply_text(f"✅ Successfully saved {source_name} to knowledge base!")
    except Exception as e:
        logging.error(f"Error saving document: {e}")
        await update.message.reply_text(f"❌ Error saving document: {str(e)}")

# ---------------------------------------------------------
# Handlers
# ---------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "မင်္ဂလာပါ! ကျွန်တော်က သင်၏ AI Secretary ပါ။ 🤖\n\n"
        "1. **PDF ဖိုင်** ပို့ပေးပါ - ကျွန်တော် ဖတ်ပြီး မှတ်ထားပါမယ်။\n"
        "2. **Website Link** ပို့ပါ - ကျွန်တော် ဖတ်ပြီး မှတ်ထားပါမယ်။\n"
        "3. **မေးခွန်းမေးပါ** - မှတ်ထားတဲ့ အချက်အလက်တွေထဲက ပြန်ဖြေပေးပါမယ်။"
    )

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle PDF file uploads"""
    document = update.message.document
    if document.mime_type != 'application/pdf':
        await update.message.reply_text("Please send a PDF file.")
        return

    await update.message.reply_text("📥 Receiving PDF...")
    
    # Download file
    file = await context.bot.get_file(document.file_id)
    
    # Use temp file to process
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
        await file.download_to_drive(custom_path=temp_pdf.name)
        
        # Load PDF
        loader = PyPDFLoader(temp_pdf.name)
        pages = loader.load()
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(pages)
        
        # Add metadata (source name)
        for doc in texts:
            doc.metadata = {"source": document.file_name}

        # Save to Supabase
        await process_document(update, context, texts, document.file_name)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages (Links or Questions)"""
    user_text = update.message.text

    # 1. Check if it's a URL
    if user_text.startswith("http://") or user_text.startswith("https://"):
        await update.message.reply_text(f"🔗 Reading website: {user_text}...")
        try:
            loader = WebBaseLoader(user_text)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(docs)
            
            # Add metadata
            for doc in texts:
                doc.metadata = {"source": user_text}

            await process_document(update, context, texts, "Website")
        except Exception as e:
            await update.message.reply_text(f"❌ Error reading website: {str(e)}")
        return

    # 2. Assume it's a Question (RAG)
    await update.message.reply_chat_action("typing")
    
    try:
        # Search Supabase
        related_docs = vector_store.similarity_search(user_text, k=3)
        
        context_text = "\n\n".join([doc.page_content for doc in related_docs])
        
        if not context_text:
             # If no context found, just ask Gemini generally
            prompt = user_text
            await update.message.reply_text("No related documents found. Answering from general knowledge...")
        else:
            # Construct Prompt with Context
            prompt = f"""
            Answer the question based ONLY on the following context:
            
            {context_text}
            
            Question: {user_text}
            """
        
        # Generate Answer
        response = llm.invoke(prompt)
        await update.message.reply_text(response.content)

    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        await update.message.reply_text("❌ Sorry, something went wrong.")

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == '__main__':
    # Flask Server for Render (Dummy Server to keep port open)
    from flask import Flask
    from threading import Thread

    flask_app = Flask('')
    @flask_app.route('/')
    def home():
        return "Bot is running!"

    def run_flask():
        port = int(os.environ.get("PORT", 10000))
        flask_app.run(host='0.0.0.0', port=port)

    # Start Flask in a separate thread
    Thread(target=run_flask).start()

    # Start Bot
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_pdf)) # Handle PDFs
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_text)) # Handle Text/Links

    print("Bot is polling...")
    application.run_polling()