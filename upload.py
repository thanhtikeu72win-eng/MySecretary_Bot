import os
import glob
# Library Check: pip install langchain-google-genai langchain-community supabase pypdf docx2txt google-generativeai
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
import google.generativeai as genai

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================

GOOGLE_API_KEY="AIzaSyCxUqSySmUw8rttC3GGJQqdR49e9voiYsw"
SUPABASE_URL = "https://kwxewupfltgyczmiagkf.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt3eGV3dXBmbHRneWN6bWlhZ2tmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE4OTg3NjgsImV4cCI6MjA4NzQ3NDc2OH0.ID_uXSsRPIk3Y_rR78YwV97NQOXsdxNA9Fnj6mZVsW0"

# Boss ရဲ့ ဖိုင်လမ်းကြောင်း (Raw String r"" ကိုသုံးထားပါတယ်)
FOLDER_PATH = r"D:\TranscendSync\Legal Issue"

# ==========================================

def main():
    print(f"🚀 Starting Bulk Upload from: {FOLDER_PATH}")
    
    if not os.path.exists(FOLDER_PATH):
        print(f"❌ Error: လမ်းကြောင်း '{FOLDER_PATH}' ကို ရှာမတွေ့ပါ။")
        return

    print("⚙️ Connecting to AI & Database...")
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # 🔥 CHANGE HERE: Using YOUR specific model 'models/gemini-embedding-001'
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)
        
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return

    # Find Files
    pdf_files = glob.glob(os.path.join(FOLDER_PATH, "*.pdf"))
    docx_files = glob.glob(os.path.join(FOLDER_PATH, "*.docx"))
    all_files = pdf_files + docx_files
    
    print(f"📂 Found {len(all_files)} files.")
    
    if len(all_files) == 0:
        print("⚠️ No files found.")
        return

    # Process Each File
    for i, file_path in enumerate(all_files):
        file_name = os.path.basename(file_path)
        print(f"\n[{i+1}/{len(all_files)}] ⏳ Reading: {file_name}...")
        
        try:
            if file_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = Docx2txtLoader(file_path)
                
            pages = loader.load()
            if not pages: continue

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(pages)
            
            for doc in texts: doc.metadata = {"source": file_name}
                
            print(f"📤 Uploading {len(texts)} chunks...")
            
            vector_store = SupabaseVectorStore.from_documents(
                documents=texts,
                embedding=embeddings,
                client=supabase,
                table_name="documents",
                query_name="match_documents"
            )
            print(f"✅ Success: {file_name}")
            
        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")

    print("\n🎉 All Done! Supabase မှာ သွားစစ်ကြည့်လို့ ရပါပြီ။")

if __name__ == "__main__":
    main()