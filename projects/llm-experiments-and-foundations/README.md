# 🚀 Enterprise Graph RAG Platform (Self-Healing LLM System)

## Overview

This project implements an **enterprise-grade Retrieval-Augmented Generation (RAG) platform**, going far beyond a simple academic pipeline.

The system is designed to simulate **real production scenarios**, with a strong focus on:

- Answer quality and traceability  
- Hallucination reduction  
- Modular and extensible architecture  
- Observability and debuggability  
- Automatic failure recovery (*self-healing*)  
- Relational reasoning via knowledge graphs (*Graph RAG*)  

The result is a **robust hybrid RAG system**, combining **vector search, reranking, knowledge graphs, review agents, and automatic correction loops**.

---

## 🧠 High-Level Architecture

```
Client
  ↓
FastAPI (API Layer)
  ↓
Retrieval Pipeline
  ├── Embeddings (Sentence Transformers)
  ├── FAISS Vector Store
  ├── Two-Stage Retrieval (Dense + Reranking)
  ↓
Graph RAG (Knowledge Graph)
  ↓
LLM Inference (Model Router)
  ↓
Review Agent (Anti-Hallucination)
  ↓
Self-Healing Agent (Automatic Retry)
  ↓
Final Answer + Sources
```

---

## 🔑 Implemented Features

### ✅ Retrieval-Augmented Generation (RAG)
- Controlled document chunking  
- Vector indexing with **FAISS**  
- Efficient semantic search  
- Metadata preservation for traceability  

### ✅ Two-Stage Retrieval (Reranking)
- Initial dense retrieval with FAISS  
- Re-ranking with **Cross-Encoder (MS MARCO)**  
- Significant improvement in contextual precision  

### ✅ Graph RAG (Knowledge Graph)
- Entity extraction (Person, Role, Technologies)  
- Dynamic **knowledge graph** construction  
- Explicit relations (`HAS_ROLE`, `WORKED_WITH`)  
- Graph-based context enrichment for the LLM  
- Dedicated API endpoint for direct graph queries  

### ✅ Review Agent (Anti-Hallucination)
- Automatic answer evaluation  
- Criteria:
  - Faithfulness  
  - Completeness  
  - Hallucination detection  
- **Deterministic and safe** implementation  

### ✅ Self-Healing RAG
- Automatic correction loop when an answer is rejected  
- Strategies:
  - Increasing `top_k`  
  - Stricter prompt constraints  
- Single retry only (prevents infinite loops)  

### ✅ Graph Visualization
- PNG visualization of the knowledge graph  
- Implemented with **NetworkX + Matplotlib**  
- Useful for debugging, validation, and explainability  

### ✅ Enterprise-Ready API
- Built with **FastAPI**  
- Automatic Swagger / OpenAPI docs  
- Clean and decoupled endpoints  

---

## 📡 API Endpoints

### `/v1/query`
RAG + Graph RAG + Self-Healing query endpoint

```json
POST /v1/query
{
  "query": "What is the name of the software engineer?",
  "top_k": 5
}
```

### `/v1/graph/query`
Direct knowledge graph query

```json
POST /v1/graph/query
{
  "entity": "José da Silva",
  "depth": 2
}
```

---

## 🧪 Evaluation & Quality

- Clear separation between:
  - Retrieval  
  - Generation  
  - Judgment  
- Ready for integration with automated evaluation metrics (*evals*)  

---

## 🧩 Tech Stack

- Python 3.10  
- FastAPI  
- FAISS  
- Sentence-Transformers  
- HuggingFace Transformers  
- Cross-Encoder  
- NetworkX  
- Matplotlib  
- PyTorch  

---

## 🐳 Docker Execution (Recommended)

The project is fully **dockerized**, ensuring reproducibility and easy deployment.

### Prerequisites
- Docker Engine  
- Docker Compose v2 (plugin)

Install Docker Compose v2 (Ubuntu):

```bash
sudo apt update
sudo apt install docker-compose-plugin
```

### Build & Run

From the `projects` directory:

```bash
docker compose build
docker compose up
```

API will be available at:

```
http://localhost:8000/docs
```

---

## 🚀 Local Execution (Without Docker)

```bash
pip install -r requirements.txt
python src/app.py        # Build index
uvicorn src.api.main:app # Run API
```

---

## 📈 Possible Extensions

- Persisting the graph in Neo4j  
- Full LLM-as-a-Judge evaluation  
- GPU-enabled containers (NVIDIA)  
- Observability (logs and metrics)  
- Web UI for graph visualization  

---

## 💼 Professional Context

This project was built with **enterprise environments** in mind, addressing real-world challenges of LLM-based systems:

- Hallucination  
- Insufficient context  
- Unstructured data  
- Explainability requirements  

It demonstrates **applied AI engineering**, not just model usage.

---

## 👤 Author

**José Geraldo do Espírito Santo Júnior**  
Brazil  

🔗 LinkedIn: https://www.linkedin.com/in/josejunior13/
