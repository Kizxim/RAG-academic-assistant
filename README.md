# Academic RAG Assistant

A local RAG chatbot for querying academic lecture materials using document ingestion, vector search, and a local LLM.

## Features

- Load lecture documents from a local folder
- Split documents into smaller text chunks
- Generate multilingual embeddings using Hugging Face models
- Store and search document vectors with Chroma
- Answer questions using Ollama with retrieved context
- Show source documents for retrieved answers

## Tech Stack

- Python
- LangChain
- Chroma
- Hugging Face Embeddings
- Ollama
- Unstructured

## Project Structure

```text
.
├── ingest.py          # Load documents, split text, create vector database
├── chat.py            # Ask questions using retrieval + local LLM
├── requirements.txt   # Python dependencies
├── README.md
└── .gitignore
```

## How It Works

Documents are loaded from a local resource folder, split into chunks, embedded into vectors, and stored in Chroma. When the user asks a question, the system retrieves relevant chunks and sends them to a local Ollama model to generate an answer with source references.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the vector database:

```bash
python ingest.py
```

Start chatting:

```bash
python chat.py
```

## Notes

- Lecture files and Chroma database are not included in this repository.
- This project is designed for local academic use and learning purposes.

