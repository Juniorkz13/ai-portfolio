
# RAG System with FastAPI, FAISS and Local LLM

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system designed to answer questions **strictly based on provided documents**, without hallucinations or external knowledge.

It combines:
- **FastAPI** for serving a REST API
- **FAISS** for efficient vector similarity search
- **Sentence Transformers** for text embeddings
- **A local Large Language Model (LLM)** (TinyLlama) for controlled text generation

The system is fully local and suitable for **production-ready AI pipelines**, **enterprise use cases**, and **privacy-sensitive environments**.

---

## Architecture

The solution follows a modular RAG architecture:

```
User Question
      ↓
Embedding Model (Sentence-Transformers)
      ↓
FAISS Vector Store (Similarity Search)
      ↓
Relevant Context Retrieval
      ↓
Local LLM (TinyLlama)
      ↓
Final Answer (Context-Grounded)
```

---

## Key Features

- 🔍 **Semantic Search** using FAISS
- 🧠 **Context-grounded answers only**
- 🚫 **Hallucination-safe prompt design**
- ⚡ **FastAPI REST interface**
- 🧩 **Modular and extensible codebase**
- 🔒 **Fully local execution (no external APIs required)**

---

## API Endpoints

### Ingest Documents
Indexes documents located in `data/raw`.

```
POST /ingest
```

### Ask Questions
Answers questions based only on indexed documents.

```
POST /ask
{
  "question": "What is this project about?"
}
```

---

## Prompt Strategy

The system enforces strict answering rules:

- Answers **must be derived exclusively from retrieved context**
- No paraphrasing or external knowledge
- If the answer is not explicitly found, the model returns:

```
"Não sei responder com base nos documentos disponíveis."
```

This ensures **trustworthy and auditable responses**, a critical requirement in real-world AI systems.

---

## Technologies Used

- Python
- FastAPI
- FAISS
- Hugging Face Transformers
- Sentence-Transformers
- PyTorch
- TinyLlama (Local LLM)

---

## Use Cases

- Internal knowledge bases
- Enterprise document Q&A
- AI assistants with controlled outputs
- LLM experimentation with low hallucination risk
- Privacy-first AI systems

---

## Project Structure

```
projects/05_llms/
│
├── data/
│   └── raw/          # Source documents
│
├── src/
│   ├── api/          # FastAPI application
│   ├── rag/          # RAG pipeline components
│   └── llm/          # Local LLM wrapper
│
├── requirements.txt
└── README.md
```

---

## Author

**José Geraldo do Espirito Santo Júnior**  
📍 Brazil  
🔗 LinkedIn: https://www.linkedin.com/in/josejunior13/

---

This project is part of a professional AI portfolio focused on **applied Machine Learning, LLM systems, and production-ready AI solutions**.
