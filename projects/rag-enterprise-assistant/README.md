# RAG Enterprise Assistant

Chatbot de atendimento empresarial baseado em documentos PDF, utilizando
Retrieval-Augmented Generation (RAG).

## 🔍 Funcionalidade
- Ingestão de documentos PDF
- Chunking e normalização de texto
- Embeddings locais com GPU
- Busca semântica com FAISS
- Geração de respostas com Google Gemini API
- Respostas restritas ao conteúdo dos documentos

## 🧠 Arquitetura
PDF → Chunking → Embeddings (local) → FAISS → Recuperação de contexto → LLM (Gemini)

## 🛠️ Tecnologias
- Python
- SentenceTransformers
- FAISS
- Google Gemini API
- PyPDF
- Torch (CUDA)

## 🚀 Objetivo
Projeto educacional e de portfólio para demonstrar implementação prática de RAG
em um cenário de atendimento empresarial.