# LLM RAG Document Assistant

## 📌 Overview
This project implements an **Intelligent Assistant based on Large Language Models (LLMs)** using the **Retrieval-Augmented Generation (RAG)** architecture.  
The system allows users to ask natural language questions about documents (PDF/TXT) and receive accurate, contextualized answers grounded in the provided content.

The project was developed **100% with open-source technologies**, without the use of paid APIs, and is architected to scale to enterprise environments.

---

## 🎯 Project Objective
Demonstrate, in a practical and professional way, the application of **LLMs in real-world scenarios**, combining:

- Natural Language Processing (NLP)
- Semantic vector search
- Prompt engineering
- Modern applied AI architecture

This project is well-suited to showcase skills for **Machine Learning Engineer, AI Engineer, and Data Scientist** roles.

---

## 🧠 Architecture (RAG)

Simplified workflow:

1. Document upload (PDF/TXT)
2. Text extraction and cleaning
3. Text chunking
4. Embedding generation
5. Storage in a vector database (Chroma)
6. Semantic retrieval
7. Answer generation using an LLM

---

## 🛠 Technologies Used

- **LLM:** Hugging Face (open-source models)
- **Embeddings:** sentence-transformers
- **Vector Database:** ChromaDB
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Language:** Python 3.10+
- **Containerization:** Docker

---

## 📂 Project Structure

```
llm-rag-document-assistant/
├── data/
│   └── raw/
├── notebooks/
├── src/
│   ├── api/
│   ├── rag/
│   ├── llm/
│   └── config.py
├── tests/
├── requirements.txt
├── Dockerfile
├── README.md
└── .env.example
```

---

## 🔐 Privacy and Cost
- **100% local execution**
- No data sent to external services
- Zero execution cost

---

## 🚀 Next Steps
- RAG pipeline implementation
- REST API development
- Streamlit interface creation
- Dockerization and final documentation

---

## 👨‍💻 Author
**José Geraldo do Espírito Santo Júnior**  
📍 Location: Brazil
