# OS Textbook RAG System

An agentic Retrieval-Augmented Generation (RAG) system that answers 
questions about Operating Systems using the Silberschatz OS Concepts 
textbook as its knowledge base.

## What It Does

Ask it any OS question. It retrieves the most relevant passages from 
the textbook, feeds them to an LLM with conversation history, and 
returns a clear explanation at senior high school level — with 
real-world analogies for every concept.

## Architecture

```
User Question
     │
     ▼
┌─────────────────────┐
│   Agentic Router    │  ← decides which tool to use
└─────────────────────┘
     │
     ├── Tool 1: Textbook Retrieval (ChromaDB vector search)
     └── Tool 2: Term Definition (quick-lookup + extensible)
     │
     ▼
┌─────────────────────┐
│   Prompt Builder    │  ← context + conversation history
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   Groq LLM          │  ← llama-3.1-8b-instant
│   (llama-3.1-8b)    │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  PostgreSQL         │  ← persistent chat history
│  Chat History       │  ← falls back to in-memory if no DB
└─────────────────────┘
```

## Tech Stack

- **LangChain** — document loading, chunking, retrieval pipeline
- **ChromaDB** — local vector database for document embeddings
- **HuggingFace Sentence Transformers** — all-MiniLM-L6-v2 embeddings (free, local)
- **Groq** — fast LLM inference (llama-3.1-8b-instant)
- **PostgreSQL** — persistent conversation history
- **Python** — core language

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/justtokins/RAG_OS
cd RAG_OS
pip install -r requirements.txt
```

### 2. Environment variables

Create a .env file:

```env
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=postgresql://user:password@localhost:5432/RAG_OS
```

Get a free Groq API key at: https://console.groq.com

DATABASE_URL is optional — the system falls back to in-memory 
history if PostgreSQL is not configured.

### 3. PostgreSQL setup (optional)

```sql
CREATE DATABASE os_rag;
```

The table is created automatically on first run.

### 4. Ingest the textbook

```bash
python ingest_pdf.py path/to/operating-system-concepts.pdf
```

This embeds the textbook and saves the vector store to ./OS_base.
Only needs to be run once.

### 5. Start the chatbot

```bash
python main.py
```

## Key Concepts Demonstrated

| RAG architecture | ChromaDB retrieval + LLM generation |
| Vector embeddings | all-MiniLM-L6-v2 384-dim sentence vectors |
| Semantic search | Cosine similarity in vector space |
| Agentic routing | Tool selection based on query type |
| Persistent memory | PostgreSQL chat history |
| Context management | Rolling window of last 3 exchanges |
| Prompt engineering | Structured prompt with context + history |

## Project Structure

```
├── ingest_pdf.py     # Load PDF, chunk, embed, store in ChromaDB
├── query.py          # Retrieval, agentic routing, LLM, chat history
|── agent.py
|── database.py
|── tools.py
├── requirements.txt  # Dependencies
├── .env.example      # Environment variable template
└── README.md
```

## Extending This System


**Replace rule-based routing with LangChain ReAct agent:**
```python
from langchain.agents import AgentExecutor, create_react_agent
# LLM decides which tool to call — fully agentic
```


## Author

Warmate Tamuno-Tokini  
Research Assistant, Rivers State University  
[LinkedIn](https://linkedin.com/in/warmate-tokins)
