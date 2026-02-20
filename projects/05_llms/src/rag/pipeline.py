from typing import List, Dict, Optional

from src.rag.embeddings import EmbeddingService
from src.rag.vector_store import FaissVectorStore



class RetrievalPipeline:
    """
    Orchestrates the retrieval stage of a RAG system.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: FaissVectorStore,
        top_k: int = 5,
        reranker: Optional[object] = None,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.top_k = top_k
        self.reranker = reranker

    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        """
        # 1. Embed query
        query_embedding = self.embedding_service.embed_query(query)

        # 2. Dense retrieval
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )

        # 3. Optional reranking
        if self.reranker:
            results = self.reranker.rerank(query, results)

        return results

    def build_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Assemble retrieved documents into a single context string.
        """
        context_chunks = []

        for doc in retrieved_docs:
            content = doc["metadata"].get("content", "")
            context_chunks.append(content)

        return "\n\n".join(context_chunks)
