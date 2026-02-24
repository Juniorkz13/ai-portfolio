from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# RAG core
from src.rag.embeddings import EmbeddingService
from src.rag.vector_store import FaissVectorStore
from src.rag.pipeline import RetrievalPipeline
from src.rag.reranker import CrossEncoderReranker

# LLM
from src.llm.model import ModelRouter

# Review Agent
from src.agents.review_agent import ReviewAgent

# Graph RAG
from src.graph.graph_builder import build_graph_from_docs

# Self-Healing Agent
from src.agents.self_healing_agent import SelfHealingAgent


# ------------------------------------------------------------------
# App config
# ------------------------------------------------------------------

app = FastAPI(
    title="Enterprise Graph RAG API",
    version="2.0.0",
    description="Production-ready RAG with FAISS, Reranking, Graph RAG and Review Agent",
)

# ------------------------------------------------------------------
# Singleton services
# ------------------------------------------------------------------

self_healing_agent = SelfHealingAgent()

embedding_service = EmbeddingService()

vector_store = FaissVectorStore(embedding_dim=384)
vector_store.load()

reranker = CrossEncoderReranker(top_n=3)

pipeline = RetrievalPipeline(
    embedding_service=embedding_service,
    vector_store=vector_store,
    top_k=5,
    reranker=reranker,
)

model_router = ModelRouter()
review_agent = ReviewAgent()

# ------------------------------------------------------------------
# API Schemas
# ------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    task: Optional[str] = "qa"


class Source(BaseModel):
    score: float
    metadata: dict
    rerank_score: Optional[float] = None


class ReviewResult(BaseModel):
    faithful: bool
    complete: bool
    hallucination: bool
    issues: str
    final_verdict: str


class QueryResponse(BaseModel):
    answer: str
    review: ReviewResult
    sources: List[Source]


class GraphQueryRequest(BaseModel):
    entity: str
    depth: Optional[int] = 1


class GraphQueryResponse(BaseModel):
    entity: str
    relations: List[str]


# ------------------------------------------------------------------
# API Endpoint — RAG + Graph + Self-Healing
# ------------------------------------------------------------------

@app.post("/v1/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Main Graph RAG endpoint.
    """

    # --------------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------------
    pipeline.top_k = request.top_k
    retrieved_docs = pipeline.retrieve(request.query)

    if not retrieved_docs:
        return {
            "answer": "O contexto recuperado não contém informações suficientes para responder a pergunta.",
            "review": {
                "faithful": True,
                "complete": True,
                "hallucination": False,
                "issues": "",
                "final_verdict": "approve",
            },
            "sources": [],
        }

    # --------------------------------------------------------------
    # Text context
    # --------------------------------------------------------------
    text_context = pipeline.build_context(retrieved_docs)

    # --------------------------------------------------------------
    # Semantic guardrail (question vs context)
    # --------------------------------------------------------------
    lower_q = request.query.lower()
    lower_ctx = text_context.lower()

    if "empresa" in lower_q and "empresa" not in lower_ctx:
        return {
            "answer": "O contexto fornecido não contém informações suficientes para responder a esta pergunta.",
            "review": {
                "faithful": True,
                "complete": True,
                "hallucination": False,
                "issues": "Context does not mention company information",
                "final_verdict": "approve",
            },
            "sources": retrieved_docs,
        }

    # --------------------------------------------------------------
    # Graph RAG context
    # --------------------------------------------------------------
    graph = build_graph_from_docs(retrieved_docs)

    graph_context = ""
    if graph.graph.number_of_edges() > 0:
        graph_context = "\n".join(
            f"{u} -[{d['relation']}]-> {v}"
            for u, v, d in graph.graph.edges(data=True)
        )

    # --------------------------------------------------------------
    # Prompt
    # --------------------------------------------------------------
    prompt = f"""
You are an AI assistant specialized in question answering over retrieved documents.

Instructions:
- Answer strictly based on the provided contexts.
- Do not add information that is not explicitly present.
- Prefer clear and complete answers.

Text Context:
{text_context}

Graph Context (relations):
{graph_context}

Question:
{request.query}

Answer:
"""

    # --------------------------------------------------------------
    # LLM inference
    # --------------------------------------------------------------
    answer = model_router.run(
        prompt=prompt,
        task=request.task,
    )

    # --------------------------------------------------------------
    # Review (anti-hallucination)
    # --------------------------------------------------------------
    review = review_agent.review(
        question=request.query,
        context=text_context + " " + graph_context,
        answer=answer,
    )

    # --------------------------------------------------------------
    # Self-healing loop (single retry)
    # --------------------------------------------------------------
    if review["final_verdict"] == "reject":
        healed_answer, healed_review, healed_docs = self_healing_agent.heal(
            question=request.query,
            pipeline=pipeline,
            model_router=model_router,
            review_agent=review_agent,
            task=request.task,
        )

        if healed_answer and healed_review["final_verdict"] == "approve":
            answer = healed_answer
            review = healed_review
            retrieved_docs = healed_docs

    # --------------------------------------------------------------
    # Response
    # --------------------------------------------------------------
    return {
        "answer": answer,
        "review": review,
        "sources": retrieved_docs,
    }


# ------------------------------------------------------------------
# API Endpoint — Graph Query
# ------------------------------------------------------------------

@app.post("/v1/graph/query", response_model=GraphQueryResponse)
def query_graph(request: GraphQueryRequest):
    """
    Query the knowledge graph directly.
    """

    retrieved_docs = pipeline.retrieve(request.entity)

    if not retrieved_docs:
        return {
            "entity": request.entity,
            "relations": [],
        }

    graph = build_graph_from_docs(
        retrieved_docs,
        root_entity=request.entity,
    )

    target = None
    request_norm = request.entity.strip().lower()

    for node in graph.graph.nodes:
        if node.strip().lower() == request_norm:
            target = node
            break

    if not target:
        return {
            "entity": request.entity,
            "relations": [],
        }

    subgraph = graph.get_subgraph(target, depth=request.depth)

    relations = [
        f"{u} -[{d['relation']}]-> {v}"
        for u, v, d in subgraph.edges(data=True)
    ]

    return {
        "entity": target,
        "relations": relations,
    }