import logging
from typing import Protocol
from typing_extensions import TypedDict

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.core.config import settings
from app.models.chunk import Chunk
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError


class RetrievalError(Exception):
    """Raised when retrieval pipeline fails."""


class RetrievalResult(TypedDict):
    """Structured chunk payload returned by semantic retrieval."""

    content: str
    document_title: str
    document_type: str
    version: str
    page_number: int
    chunk_index: int
    document_id: int


class RetrievalFilters(TypedDict, total=False):
    """Optional filters applied before vector similarity ranking."""

    document_id: int
    document_type: str
    version: str


class QuestionEmbedder(Protocol):
    """Contract for question embedding generation."""

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding vector for input text."""


class RetrievalService:
    """Retrieve semantically similar chunks from pgvector-enabled storage."""

    def __init__(
        self,
        db: Session,
        embedding_service: QuestionEmbedder | None = None,
    ):
        self.db = db
        self.embedding_service = embedding_service or EmbeddingService()
        self.embedding_dimension = settings.embedding_dimension
        self.logger = logging.getLogger(__name__)

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the top-k most relevant chunks for a natural language question."""
        self.logger.info(
            "Retrieval request received",
            extra={"question_length": len(question or ""), "top_k": top_k, "has_filters": bool(filters)},
        )
        if not question or not question.strip():
            raise ValueError("question must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        try:
            self.logger.info("Generating question embedding")
            question_embedding = self.embedding_service.embed_text(question)
            self.logger.info(
                "Question embedding generated",
                extra={"embedding_dimensions": len(question_embedding)},
            )
            if len(question_embedding) != self.embedding_dimension:
                raise RetrievalError(
                    "Question embedding dimension mismatch for vector search."
                )
        except RetrievalError:
            self.logger.exception("Question embedding dimension mismatch")
            raise
        except (EmbeddingServiceError, ValueError) as exc:
            self.logger.exception("Failed to generate question embedding")
            raise RetrievalError("Failed to embed question for retrieval.") from exc

        try:
            self.logger.info("Querying vector database for similar chunks", extra={"top_k": top_k})
            stmt = select(Chunk).options(joinedload(Chunk.document)).where(Chunk.embedding.is_not(None))
            if filters:
                if filters.get("document_id") is not None:
                    stmt = stmt.where(Chunk.document_id == filters["document_id"])
                if filters.get("document_type"):
                    stmt = stmt.where(Chunk.document.has(document_type=filters["document_type"]))
                if filters.get("version"):
                    stmt = stmt.where(Chunk.document.has(version=filters["version"]))
            stmt = stmt.order_by(Chunk.embedding.cosine_distance(question_embedding)).limit(top_k)
            chunks = self.db.scalars(stmt).all()
            self.logger.info("Retrieved chunks from vector database", extra={"retrieved_chunks": len(chunks)})
        except Exception as exc:
            self.logger.exception("Vector similarity query failed")
            raise RetrievalError("Failed to query similar chunks from vector store.") from exc

        self.logger.info("Retrieval request completed successfully")
        return [
            {
                "content": chunk.content,
                "document_title": chunk.document.title if chunk.document else "Unknown document",
                "document_type": chunk.document.document_type if chunk.document else "unknown",
                "version": chunk.document.version if chunk.document else "unknown",
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "document_id": chunk.document_id,
            }
            for chunk in chunks
        ]
