from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
import os
import certifi


# Ensure Python SSL uses certifi's certificate bundle.
# Necessary for the Ollama Python client on some Windows setups.

os.environ["SSL_CERT_FILE"] = certifi.where()

# 1. Load the same embedding model used during the build phase

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load the existing, persisted Chroma vector store

DB_DIR = "./chroma_db"

vector_store = Chroma(
    collection_name="ak_collection",
    embedding_function=embedding_model,   # must match build phase
    persist_directory=DB_DIR
)

print("[INFO] Vector store loaded.")
# 3. Get user query

query = input("Ask a question: ")
# 4. Retrieve top-k relevant chunks from Chroma

results = vector_store.similarity_search(query, k=3)

# Combine retrieved chunks into a single context string
context = "\n\n---\n\n".join([doc.page_content for doc in results])

print("\n[DEBUG] Retrieved Context:\n", context)

# 5. Initialize local LLM (Ollama Mistral)

llm = ChatOllama(
    model="mistral",     # must match your locally-installed Ollama model
    temperature=0.0,     # deterministic responses
    num_predict=512      # max tokens to generate
)
# 6. Construct messages (system + human) for Mistral

messages = [
    (
        "system",
        "Answer ONLY using the context provided below. "
        "If the answer is not present, respond with: 'I don't know'."
    ),
    (
        "human",
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
]

# 7. Generate the response from the LLM

response = llm.invoke(messages)

# 8. Display final answer

print("\n=== ANSWER ===\n")
print(response.content)
