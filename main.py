#!/usr/bin/env python3
"""
main.py
Single-file RAG pipeline (build + query) for the AmbedkarGPT intern task.

Usage:
  # Normal (build if needed, then query)
  python main.py

  # Force rebuild the vector DB (delete existing DB and recreate)
  python main.py --rebuild
"""

import os
import shutil
import argparse
import certifi

# Ensure SSL uses certifi's CA bundle before any httpx/ollama import
os.environ["SSL_CERT_FILE"] = certifi.where()

# Core LangChain imports (document loader, splitter, embeddings, vectorstore)
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# LangChain Ollama wrapper (local LLM). Ollama CLI + model must be running locally.
from langchain_ollama import ChatOllama


# ------------------------
# Config
# ------------------------
DB_DIR = "./chroma_db"                      # where Chroma persists data
COLLECTION_NAME = "ak_collection"           # logical collection name
SPEECH_FILE = "speech.txt"                  # input file (provided)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
TOP_K = 3                                   # number of chunks to retrieve for each query


def build_vector_store(force_rebuild: bool = False):
    """
    Build the Chroma vector store from the input text.
    If `force_rebuild` is True, the existing DB folder is deleted first.
    If DB_DIR already exists and force_rebuild is False, building is skipped.
    Returns a Chroma vector_store instance (loaded and ready).
    """
    # If requested, delete old DB to avoid duplication (useful during development)
    if force_rebuild and os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        print("[INFO] Old Chroma database deleted (forced rebuild).")

    # If DB exists and not forcing rebuild, simply load and return it.
    if os.path.exists(DB_DIR):
        print("[INFO] Existing Chroma DB found — loading (skip build).")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vs = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=DB_DIR,
        )
        return vs


    # 1) Load the input text

    if not os.path.exists(SPEECH_FILE):
        raise FileNotFoundError(f"Could not find input file: {SPEECH_FILE}")

    loader = TextLoader(SPEECH_FILE, encoding="utf-8")
    documents = loader.load()
    text = documents[0].page_content
    print("[INFO] Loaded input text from", SPEECH_FILE)

    # 2) Split into chunks
 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # recommended split order: paragraph -> line -> space -> char
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.create_documents([text])

    # Add metadata (source & index) for traceability
    metadatas = []
    for i, c in enumerate(chunks):
        metadatas.append({"source": SPEECH_FILE, "chunk_index": i})
    # Replace chunks with ones that include metadata (create_documents supports metadatas)
    # Note: create_documents already returned Documents; we will add metadata manually if needed.
    chunks = text_splitter.create_documents([text], metadatas=[{"source": SPEECH_FILE, "chunk_index": i} for i in range(len(chunks))])
    print(f"[INFO] Split text into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    # 3) Embedding model
 
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("[INFO] Initialized embedding model:", EMBEDDING_MODEL_NAME)


    # 4) Create & populate Chroma
  
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIR,
        embedding_function=embedding_model,
    )

    # Add documents to the vector store (Chroma computes embeddings internally)
    result = vector_store.add_documents(chunks)
    print("[INFO] Added documents to Chroma. Addition result:", result)
    print("[INFO] Vector store created and persisted at:", DB_DIR)

    return vector_store


def query_loop(vector_store):
    """
    Repeatedly ask user for a query, retrieve top-k chunks, call Ollama, and print answer.
    """
    # Initialize local Ollama chat model (assumes ollama daemon is running and 'mistral' model is available)
    llm = ChatOllama(model="mistral", temperature=0.0, num_predict=512)

    print("\n[READY] Ask questions about the speech (type 'exit' or 'quit' to stop).")
    while True:
        query = input("\nAsk a question: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("[INFO] Exiting.")
            break

        # Retrieve relevant chunks (vector similarity)
        results = vector_store.similarity_search(query, k=TOP_K)
        if not results:
            print("[WARN] No results returned from vector store.")
            continue

        # Combine retrieved texts into a single context for the LLM
        context = "\n\n---\n\n".join([doc.page_content for doc in results])
        # Optional: include metadata (sources) in debug display
        sources = [doc.metadata if hasattr(doc, "metadata") else {} for doc in results]
        print("\n[DEBUG] Retrieved sources:", sources)
        print("\n[DEBUG] Context passed to LLM:\n", context)

        # Build system + human messages (explicit instruction to use only the provided context)
        system_message = (
            "You are an assistant. Answer the user's question using ONLY the provided context. "
            "If the answer is not contained in the context, respond with: 'I don't know.' Be concise."
        )
        human_message = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        # Call the local Ollama model
        try:
            messages = [("system", system_message), ("human", human_message)]
            response = llm.invoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)
            print("\n=== ANSWER ===\n")
            print(answer)
        except Exception as e:
            print("[ERROR] LLM invocation failed:", str(e))
            print("Make sure Ollama is installed, the daemon is running, and the 'mistral' model is available.")
            print("You can run: ollama pull mistral  # and then ollama serve or `ollama run mistral` as needed.")


def main():
    parser = argparse.ArgumentParser(description="Single-file RAG pipeline (build + query)")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the Chroma DB (delete existing DB_dir first).")
    args = parser.parse_args()

    # Build (or load) the vector store
    vector_store = build_vector_store(force_rebuild=args.rebuild)

    # Enter query loop (ask questions and get answers)
    query_loop(vector_store)


if __name__ == "__main__":
    main()
