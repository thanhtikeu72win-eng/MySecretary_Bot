import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import create_client

# Load environment variables
load_dotenv()

# Setup Supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

# Setup Embeddings (Updated Model)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def ingest_docs():
    print("🚀 Loading documents...")
    loader = DirectoryLoader("knowledge_base/", glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    print(f"📄 Loaded {len(documents)} documents.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    print(f"✂️ Split into {len(docs)} chunks.")
    
    print("💾 Uploading to Supabase (this may take a while)...")
    vector_store = SupabaseVectorStore.from_documents(
        docs,
        embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents",
        chunk_size=500
    )
    
    print("✅ Ingestion Complete!")

if __name__ == "__main__":
    ingest_docs()