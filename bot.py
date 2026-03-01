import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone # For direct deletion

# 1. Load Environment Variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "legal-bot"

# 2. Setup Page Config
st.set_page_config(page_title="Legal AI Assistant", layout="wide")

# 3. Sidebar (My Brain Panel & Tools)
with st.sidebar:
    st.header("🧠 My Brain Panel")
    
    # --- Status Indicators ---
    if GOOGLE_API_KEY:
        st.success("✅ Google AI Ready")
    else:
        st.error("❌ Google AI Key Missing")
        
    if PINECONE_API_KEY:
        st.success("✅ Database Connected")
    else:
        st.error("❌ Database Key Missing")

    st.divider()

    # --- AI Tools (Features) ---
    st.subheader("🛠️ AI Tools")
    tool_choice = st.radio(
        "Choose a Tool:",
        ["💬 Chat (Default)", "✉️ Email Draft", "📝 Summarize", "🇬🇧⇄🇲🇲 Translate", "🧾 Report"]
    )

    st.divider()
    
    # --- 🗑️ DELETE DATA SECTION (NEW) ---
    with st.expander("🗑️ Delete Data (Danger Zone)", expanded=False):
        st.warning("Warning: This action cannot be undone!")
        delete_source = st.text_input("Paste Source Path to Delete:", placeholder="e.g., D:\\TranscendSync\\File.docx")
        
        if st.button("🗑️ Delete File Data"):
            if delete_source:
                try:
                    # Initialize Pinecone Client directly
                    pc = Pinecone(api_key=PINECONE_API_KEY)
                    index = pc.Index(PINECONE_INDEX_NAME)
                    
                    # Delete by Filter
                    with st.spinner("Deleting vectors..."):
                        index.delete(filter={"source": {"$eq": delete_source}})
                        st.success(f"✅ Deleted all data for: {delete_source}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            else:
                st.warning("⚠️ Please enter a source path.")

    st.divider()
    with st.expander("ℹ️ Help / Guide"):
        st.markdown("""
        - **Chat:** Ask general legal questions.
        - **Delete Data:** Paste the full file path from your logs to remove duplicates.
        """)

# 4. Main App Logic
st.title("⚖️ Legal AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Initialize AI & Database
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# 6. Define Prompts for Tools 📝
def get_prompt_by_tool(tool_name, user_input):
    base_prompt = ""
    
    if tool_name == "✉️ Email Draft":
        base_prompt = f"You are a professional lawyer. Draft a formal email about: '{user_input}'. Use polite and professional language."
    
    elif tool_name == "📝 Summarize":
        base_prompt = f"Please summarize the following legal text or topic concisely in bullet points: '{user_input}'."
    
    elif tool_name == "🇬🇧⇄🇲🇲 Translate":
        base_prompt = f"Translate the following text to the other language (English to Burmese OR Burmese to English) accurately, maintaining legal terminology: '{user_input}'."
    
    elif tool_name == "🧾 Report":
        base_prompt = f"Generate a formal legal report structure regarding: '{user_input}'. Include Introduction, Legal Analysis, and Conclusion sections."
    
    return base_prompt

# 7. Chat Input & Processing
if prompt := st.chat_input("မေးခွန်း မေးမြန်းပါ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                special_prompt = get_prompt_by_tool(tool_choice, prompt)

                if special_prompt:
                    # Tool Logic (Direct LLM)
                    response = llm.invoke(special_prompt)
                    result = response.content
                else:
                    # Chat Logic (RAG)
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(),
                    )
                    response = qa_chain.invoke(prompt)
                    result = response["result"]

                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})

            except Exception as e:
                st.error(f"Error: {e}")