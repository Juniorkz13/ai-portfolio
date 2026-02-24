# Enterprise RAG Assistant (Gemini + FAISS)

## Overview
This project implements an **enterprise-grade Retrieval-Augmented Generation (RAG) assistant** designed to simulate a real-world corporate customer support agent.

The system allows the ingestion of internal documents (PDFs), builds a semantic search index, and answers user questions in a conversational way using a Large Language Model (LLM) via API.

This repository was built as a **portfolio project**, with a strong focus on:
- clean architecture
- production-oriented decisions
- explainability for technical interviews

---

## Key Features
- 📄 PDF document ingestion
- 🧠 Semantic search with FAISS
- 🔎 Context-aware question answering (RAG)
- 💬 Conversational memory per session
- ⚡ Response caching
- ❤️ Healthcheck endpoint
- ⏱️ Rate limiting for cost protection
- 🐳 Fully Dockerized
- 🔐 Secrets managed via environment variables
- ☁️ Free-tier friendly (Gemini API)

---

## Architecture Overview

```
User
  |
  v
FastAPI (/chat)
  |
  |-- Rate Limiter
  |-- Session Memory
  |
  v
Retriever (FAISS)
  |
  v
Relevant Context Chunks
  |
  v
Prompt Builder
  |
  v
Gemini LLM (API)
  |
  v
Final Answer
```

---

## Technology Stack

### Backend
- **Python 3.10**
- **FastAPI**
- **Uvicorn**

### RAG & ML
- **Sentence-Transformers** (MiniLM)
- **FAISS (CPU)**
- **PyPDF**

### LLM
- **Google Gemini API**
  - `gemini-flash-latest` (default, configurable)

### Infrastructure
- **Docker**
- **Docker Compose**

---

## Project Structure

```
rag-enterprise-assistant/
├── app/
│   ├── main.py
│   ├── rag.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── bootstrap.py
│   ├── config.py
│   └── state.py
├── data/
│   └── raw/
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Environment Variables

Create a `.env` file based on `.env.example`:

```env
ENV=dev

GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-flash-latest

RAG_TOP_K=3
CACHE_SIZE=128
```

---

## Running Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## Running with Docker

```bash
docker compose up rag-enterprise-assistant
```

API:
- http://localhost:8001/docs

---

## API Endpoints

### POST /chat
Main conversational endpoint.

### GET /health
Healthcheck endpoint.

---

## Design Decisions
- RAG to avoid hallucinations
- FAISS for simplicity and performance
- Gemini API for zero-cost experimentation

---

## Author

**José Geraldo do Espírito Santo Júnior**  
📍 Brazil  

🔗 LinkedIn:  
https://www.linkedin.com/in/josejunior13/

---

## Final Notes
This project was designed to reflect real-world engineering trade-offs and production patterns.
