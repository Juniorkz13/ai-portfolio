# 🚀 Enterprise Graph RAG Platform (Self-Healing LLM System)

## Visão Geral

Este projeto implementa uma **plataforma de Retrieval-Augmented Generation (RAG) de nível empresarial**, indo muito além de um pipeline acadêmico simples.

O sistema foi projetado para simular **cenários reais de produção**, com foco em:

- Qualidade e rastreabilidade das respostas  
- Redução de alucinações  
- Arquitetura modular e extensível  
- Observabilidade e debug  
- Recuperação automática de falhas (*self-healing*)  
- Raciocínio relacional via grafos (*Graph RAG*)  

O resultado é um **RAG híbrido e robusto**, combinando **busca vetorial, reranking, knowledge graph, agentes de revisão e correção automática**.

---

## 🧠 Arquitetura Geral

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

## 🔑 Funcionalidades Implementadas

### ✅ Retrieval-Augmented Generation (RAG)
- Chunking controlado de documentos  
- Indexação vetorial com **FAISS**  
- Busca semântica eficiente  
- Metadados preservados para rastreabilidade  

### ✅ Two-Stage Retrieval (Reranking)
- Recuperação inicial com FAISS  
- Reordenação com **Cross-Encoder (ms-marco)**  
- Ganho real de precisão contextual  

### ✅ Graph RAG (Knowledge Graph)
- Extração de entidades (Pessoa, Cargo, Tecnologias)  
- Construção dinâmica de um **grafo de conhecimento**  
- Relações explícitas (`HAS_ROLE`, `WORKED_WITH`)  
- Uso do grafo como contexto adicional para o LLM  
- Endpoint dedicado para consulta direta ao grafo  

### ✅ Review Agent (Anti-Hallucination)
- Avaliação automática das respostas  
- Critérios:
  - Faithfulness  
  - Completeness  
  - Hallucination  
- Implementação **determinística e segura**  

### ✅ Self-Healing RAG
- Loop automático de correção quando a resposta é rejeitada  
- Estratégias:
  - Aumento do `top_k`  
  - Novo prompt mais restritivo  
- Apenas **uma nova tentativa**, evitando loops infinitos  

### ✅ Visualização do Grafo
- Geração de imagem PNG do knowledge graph  
- Implementado com **NetworkX + Matplotlib**  
- Útil para debug, validação e explicabilidade  

### ✅ API Enterprise
- Desenvolvida com **FastAPI**  
- Swagger/OpenAPI automático  
- Endpoints bem definidos e desacoplados  

---

## 📡 Endpoints

### `/v1/query`
Perguntas usando **RAG + Graph RAG + Self-Healing**

```json
POST /v1/query
{
  "query": "qual o nome do engenheiro de software?",
  "top_k": 5
}
```

### `/v1/graph/query`
Consulta direta ao knowledge graph

```json
POST /v1/graph/query
{
  "entity": "José da Silva",
  "depth": 2
}
```

---

## 🧪 Avaliação e Qualidade

- Separação clara entre:
  - Recuperação  
  - Geração  
  - Julgamento  
- Preparado para integração com métricas automáticas de *evals*  

---

## 🧩 Tecnologias Utilizadas

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

## 🐳 Execução com Docker (Recomendado)

O projeto é totalmente **dockerizado**, garantindo reprodutibilidade e facilidade de deploy.

### Pré-requisitos
- Docker Engine
- Docker Compose v2 (plugin)

Instalação do Compose v2 (Ubuntu):

```bash
sudo apt update
sudo apt install docker-compose-plugin
```

### Build e execução

Na pasta `projects`:

```bash
docker compose build
docker compose up
```

A API ficará disponível em:

```
http://localhost:8000/docs
```

### Persistência
- Índice FAISS persistido via volume  
- Cache de modelos HuggingFace reutilizado entre execuções  

---

## 🚀 Execução Local (sem Docker)

```bash
pip install -r requirements.txt
python src/app.py        # Indexação
uvicorn src.api.main:app # API
```

---

## 📈 Possíveis Evoluções

- Persistência do grafo em Neo4j  
- LLM-as-a-Judge completo  
- Containers GPU (NVIDIA)  
- Observabilidade (logs e métricas)  
- Interface web para visualização do grafo  

---

## 💼 Contexto Profissional

Este projeto foi desenvolvido com foco em **ambientes corporativos**, abordando problemas reais de sistemas baseados em LLMs:

- Alucinação  
- Contexto insuficiente  
- Dados não estruturados  
- Necessidade de explicabilidade  

Ele demonstra **engenharia de IA aplicada**, não apenas uso de modelos.

---

## 👤 Autor

Projeto desenvolvido por **Júnior Kz**  
Engenharia de IA • LLMs • RAG • Sistemas Inteligentes
