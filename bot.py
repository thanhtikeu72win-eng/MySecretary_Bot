import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from supabase import create_client, Client

# Setup Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Load Environment Variables
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Gemini AI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

# Initialize Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user.first_name
    await context.bot.send_message(
        chat_id=update.effective_chat.id, 
        text=f"မင်္ဂလာပါ {user}! \n\n/save [မှတ်ချင်တာ] - မှတ်စု သိမ်းမည်\n/list - မှတ်စု ပြန်ကြည့်မည်\n\nကျန်တာကတော့ AI နဲ့ စကားပြောလို့ ရပါတယ်!"
    )

async def save_note(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    note_content = " ".join(context.args)
    
    if not note_content:
        await update.message.reply_text("⚠️ ဘာမှတ်ရမလဲ ပြောပြပါဦး။ (ဥပမာ: /save မနက်ဖြန် Meeting ရှိသည်)")
        return

    try:
        data = {"user_id": user_id, "content": note_content}
        supabase.table("notes").insert(data).execute()
        await update.message.reply_text("✅ မှတ်ပြီးပါပြီ!")
    except Exception as e:
        print(f"DB Error: {e}")
        await update.message.reply_text("❌ သိမ်းလို့ မရဘူး ဖြစ်နေတယ်။")

async def list_notes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        response = supabase.table("notes").select("*").eq("user_id", user_id).execute()
        notes = response.data
        
        if not notes:
            await update.message.reply_text("📭 ဘာမှ မှတ်မထားပါဘူး။")
        else:
            msg = "📋 **မှတ်ထားသော စာများ:**\n"
            for note in notes:
                msg += f"- {note['content']}\n"
            await update.message.reply_text(msg)
    except Exception as e:
        print(f"DB Error: {e}")
        await update.message.reply_text("❌ ပြန်ကြည့်လို့ မရဘူး ဖြစ်နေတယ်။")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    chat_id = update.effective_chat.id
    
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    
    try:
        messages = [
            SystemMessage(content="You are a helpful AI assistant. Answer in Myanmar language."),
            HumanMessage(content=user_text)
        ]
        response = llm.invoke(messages)
        await context.bot.send_message(chat_id=chat_id, text=response.content)

    except Exception as e:
        print(f"AI Error: {e}")
        await context.bot.send_message(chat_id=chat_id, text="⚠️ AI Error")

if __name__ == '__main__':
    print("🤖 Bot Starting with Database...")
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .read_timeout(30)
        .write_timeout(30)
        .build()
    )
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('save', save_note))
    application.add_handler(CommandHandler('list', list_notes))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    print("✅ Polling started...")
    application.run_polling()
