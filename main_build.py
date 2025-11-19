import os
import shutil

# Load text into LangChain Document objects
from langchain_community.document_loaders import TextLoader

# Split long text into smaller chunks for embedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local embedding model (MiniLM) – runs offline, no API keys required
from langchain_huggingface import HuggingFaceEmbeddings

# Chroma – local vector database for storing embeddings
from langchain_chroma import Chroma



# Configuration

DB_DIR = "./chroma_db"  # Directory where Chroma will save the vector database



# Remove old database (prevents duplicate vectors during testing)

if os.path.exists(DB_DIR):
    shutil.rmtree(DB_DIR)
    print("[INFO] Old Chroma database deleted.")


# 1. Load the input text file

file_path = "speech.txt"

loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()                    # List[Document]
text = documents[0].page_content             # Extract text content

print("[INFO] Loaded speech.txt")


# 2. Split text into smaller chunks for embedding

# RecursiveCharacterTextSplitter breaks the text using a hierarchy:
# Paragraph → Line → Word → Character (as needed)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,          # Max characters per chunk
    chunk_overlap=50,        # Overlap to preserve context between chunks
    separators=["\n\n", "\n", "", " "]   # Split strategy
)

chunks = text_splitter.create_documents([text])
print(f"[INFO] Created {len(chunks)} chunks.")

# 3. Initialize the embedding model

# Using MiniLM-L6-v2 because it is fast, lightweight, and ideal for local RAG and also it as per the assignment requirement
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Create and populate the Chroma vector store

# persist_directory → enables saving the vector DB to disk
vector_store = Chroma(
    collection_name="ak_collection",
    persist_directory=DB_DIR,
    embedding_function=embedding_model,
)

# Convert chunks to embeddings and store them in Chroma
rat = vector_store.add_documents(chunks)

print("this is ---->", rat)   # Debug: shows inserted document IDs
print("[INFO] Vector Store created and persisted successfully!")
