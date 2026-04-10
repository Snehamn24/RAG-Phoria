# STEP 1: Load PDF
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"dataset/Employee-Handbook.pdf")
documents = loader.load()

# STEP 2: Split (optimized)
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

docs = splitter.split_documents(documents)

# STEP 3: Faster Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # fast model
)

# STEP 4: FAISS (Load if exists, else create)
from langchain_community.vectorstores import FAISS
import os

if os.path.exists("faiss_index"):
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")

# STEP 5: User Query (IMPORTANT)
query = input("Enter your question: ")

# STEP 6: Improved Retrieval (MMR + faster)
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4}
)

retrieved_docs = retriever.invoke(query)

# Reduce context size for speed
context = "\n".join([doc.page_content[:300] for doc in retrieved_docs])

# STEP 7: Gemma (faster variant)
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="gemma:2b-instruct")

# STEP 8: Better Prompt
prompt = f"""
You are an assistant answering from a company handbook.

Use the context to answer clearly.
Even if exact words are not present, infer from related information.

Context:
{context}

Question:
{query}

Answer:
"""

# STEP 9: Generate Answer
response = llm.invoke(prompt)

print("\nRetrieved Context:\n")
print(context)

print("\nFinal Answer:\n")
print(response)