import sys
import os

print("--- DIAGNOSTIC REPORT ---")

try:
    import langchain
    print(f"✅ LangChain Version: {langchain.__version__}")
except ImportError:
    print("❌ LangChain Not Found")

try:
    from langchain.chains import RetrievalQA
    print("✅ LangChain Chains: OK")
except ImportError as e:
    print(f"❌ LangChain Chains Error: {e}")

try:
    import langchain_community
    print(f"✅ LangChain Community Version: {langchain_community.__version__}")
except ImportError:
    print("❌ LangChain Community Not Found")
    
print("-------------------------")