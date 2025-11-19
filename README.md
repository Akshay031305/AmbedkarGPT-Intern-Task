RAG System using LangChain + ChromaDB + HuggingFace MiniLM + Ollama Mistral
A complete Retrieval-Augmented Generation pipeline implemented in Python
🚀 Overview

This project implements a full Retrieval-Augmented Generation (RAG) pipeline using:

LangChain for orchestration

ChromaDB as a local vector store (100% free & offline)

HuggingFace MiniLM-L6-v2 embeddings (lightweight, fast, no API keys)

Ollama Mistral 7B as the local LLM (free, runs on your machine)

The system takes a text file (speech.txt), breaks it into chunks, embeds them, stores them in Chroma, retrieves the most relevant chunks based on user questions, and feeds the retrieved context to the Mistral model to generate answers.

Everything runs locally, with no API keys, no cloud, and no cost.

📌 1. main.py — Full End-to-End RAG Pipeline (Build + Query)

main.py is a single-file solution that handles the entire workflow:

✔ Build Phase (Runs only when needed)

Loads speech.txt

Splits it into chunks

Generates embeddings using MiniLM

Stores embeddings in a persistent Chroma database (./chroma_db/)

✔ Query Phase (Runs every time)

Loads the previously built Chroma vector store

Accepts a user question

Retrieves the top-K relevant chunks

Sends context + question to Ollama Mistral 7B

Prints the final answer

✔ Rebuild option

You can force a fresh rebuild (delete the vector DB) using:

python main.py --rebuild

This avoids duplication and ensures clean embeddings during development.

📁 2. Production-Ready Code (main_build.py + main_query.py)

In addition to the single-file main.py, this project also includes:

🔹 main_build.py

Builds the vector database once.
This script is ideal when your data rarely changes.

🔹 main_query.py

Fast inference-only pipeline.
Loads the existing vector store and answers queries instantly.

⭐ Why this split is good in production:

Prevents accidental duplication of embeddings

Faster startup (you don’t need to split/embed every run)

Efficient deployment—embedding creation happens offline/one-time

Clean architecture—RAG best practices recommend separating build/query

Reproducible—keeps vector indexing stable and predictable

Scalable—easy to swap text files, models, or vector DBs later

This structure mirrors real-world RAG systems used in industry.

📦 Project Structure
rag-project/
│
├── main.py # Combined build + query pipeline
├── main_build.py # Production-style build script
├── main_query.py # Production-style query script
│
├── speech.txt # Input file for the RAG system
├── chroma_db/ # Auto-generated Chroma vector store
│
├── requirements.txt
└── README.md

⚙️ Installation

1. Create and activate a Python environment

(Recommended: Python 3.10)

conda create -n rag python=3.10 -y
conda activate rag

2. Install dependencies
   pip install -r requirements.txt

3. Install Ollama & Mistral locally

From https://ollama.ai

Then pull the Mistral model:

ollama pull mistral

Make sure the Ollama server is running.

▶️ How to Run
Option A — Use single file (main.py)

Best for demonstration and assignment submission.

python main.py

Force rebuild the vector DB:

python main.py --rebuild

Option B — Production Style (recommended for real projects)
Build vector store once:
python main_build.py

Query anytime:
python main_query.py

❓ How the System Works (Technical Explanation)

1. Document Loading

speech.txt is loaded and converted into a Document object by LangChain.

2. Chunking

Text is chunked using:

chunk_size = 200 characters

chunk_overlap = 50
This preserves contextual continuity.

3. Embeddings

Each chunk is embedded using:

sentence-transformers/all-MiniLM-L6-v2

Why MiniLM?

Fast

Accurate

Lightweight

Perfect for CPU-only setups

Ideal for RAG retrieval tasks

4. Vector Store (ChromaDB)

Embeddings + metadata are stored locally in:

./chroma_db/

Chroma automatically handles search indexing and persistence.

5. Retrieval

Similarity search returns the top-k chunks relevant to a user’s question.

6. LLM Answering

Retrieved context + question → fed into Ollama Mistral.
The system message forces the model to answer ONLY using context.

Conclusion

This repository provides a complete, production-ready Retrieval-Augmented Generation system using modern tooling — fully offline, lightweight, and easy to run.
"# AmbedkarGPT-Intern-Task" 
