import os
import logging
import asyncio
import tempfile
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# --- AI & Database Libraries ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client

# --- Setup Logging ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- Environment Variables ---
TOKEN = os.getenv("TELEGRAM_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Setup AI & DB Clients ---
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

# --- Bot Commands ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "မင်္ဂလာပါ! ကျွန်တော်က MySecretary AI Bot ပါ။\n\n"
        "ကျွန်တော့်ကို PDF ဖိုင်တွေ ပို့ပေးလို့ ရသလို၊ Link တွေ ပို့ပေးပြီး မှတ်ခိုင်းလို့ရပါတယ်။\n"
        "ပြီးရင် သိချင်တာ ပြန်မေးနိုင်ပါတယ်ခင်ဗျာ။"
    )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.document.get_file()
    file_name = update.message.document.file_name

    if not file_name.endswith('.pdf'):
        await update.message.reply_text("PDF ဖိုင်ပဲ လက်ခံပါတယ်ခင်ဗျာ။")
        return

    status_msg = await update.message.reply_text(f"📄 '{file_name}' ကို ဖတ်နေပါတယ်... ခဏစောင့်ပါ...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        await file.download_to_drive(tmp_file.name)
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        vector_store.add_documents(chunks)
        await status_msg.edit_text(f"✅ '{file_name}' ကို ဖတ်ပြီး မှတ်ဉာဏ်ထဲ ထည့်လိုက်ပါပြီ! သိချင်တာ မေးလို့ရပါပြီ။")
    except Exception as e:
        await status_msg.edit_text(f"❌ Error ဖြစ်သွားပါတယ်: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text

    if user_text.startswith("http"):
        status_msg = await update.message.reply_text(f"🌐 Link ကို ဖတ်နေပါတယ်...")
        try:
            loader = WebBaseLoader(user_text)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            vector_store.add_documents(chunks)
            await status_msg.edit_text(f"✅ Link ကို ဖတ်ပြီးပါပြီ!")
        except Exception as e:
            await status_msg.edit_text(f"❌ Link ဖတ်မရပါ: {str(e)}")
        return

    docs = vector_store.similarity_search(user_text)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    if not context_text:
        response = llm.invoke(user_text)
        await update.message.reply_text(response.content)
    else:
        prompt = f"""
        အောက်ပါ အချက်အလက်များကို အခြေခံပြီး မေးခွန်းကို ဖြေပါ။
        
        အချက်အလက်များ:
        {context_text}
        
        မေးခွန်း: {user_text}
        """
        response = llm.invoke(prompt)
        await update.message.reply_text(response.content)

if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    print("Bot is running...")
    application.run_polling()