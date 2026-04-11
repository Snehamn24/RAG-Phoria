# STEP 1: Imports
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

# GLOBAL VARIABLES (cache)
db = None
llm = None


# STEP 2: Initialize DB (only once)
def load_db():
    global db

    if db is not None:
        return db

    print("🔄 Loading and processing PDF...")

    # Load PDF
    loader = PyPDFLoader("dataset/Employee-Handbook.pdf")
    documents = loader.load()

    # Split (Improved chunking)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # FAISS (load or create)
    if os.path.exists("faiss_index"):
        print("📂 Loading existing FAISS index...")
        db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("⚡ Creating new FAISS index...")
        db = FAISS.from_documents(docs, embeddings)
        db.save_local("faiss_index")

    return db


# STEP 3: Initialize LLM (only once)
def load_llm():
    global llm

    if llm is None:
        print("🤖 Loading LLM...")
        llm = OllamaLLM(model="gemma:2b-instruct")

    return llm


# 🔥 MAIN FUNCTION
def get_answer(query):
    db = load_db()
    llm = load_llm()

    # 🔍 Retrieve documents (Improved retrieval)
    retrieved_docs = db.similarity_search(query, k=6)

    # 🔎 DEBUG (optional – remove later)
    print("\n===== RETRIEVED CHUNKS =====")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Chunk {i+1} ---\n{doc.page_content}")

    # 📌 IMPORTANT: NO TRUNCATION
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # 🧠 Improved Prompt
    prompt = f"""
You are an assistant answering strictly from a company handbook.

Rules:
- Answer ONLY from the given context
- Give a clear and complete answer
- Do NOT give vague or generic answers
- Do NOT say "the document states"
- If answer is not found, say: Not found in document

Context:
{context}

Question:
{query}

Answer:
"""

    # 🤖 Generate response
    response = llm.invoke(prompt)

    return response