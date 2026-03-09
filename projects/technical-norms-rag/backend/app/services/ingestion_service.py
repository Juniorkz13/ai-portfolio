import logging
from pathlib import Path
from typing import Protocol
from typing_extensions import TypedDict

from sqlalchemy.orm import Session

from app.repositories.chunk_repository import ChunkCreateInput, ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services.chunk_service import ChunkService, PDFPage
from app.services.embedding_service import EmbeddingService
from app.services.pdf_service import PDFService


class IngestionResult(TypedDict):
    """Structured response returned by the ingestion pipeline."""

    document_id: int
    total_pages: int
    total_chunks: int


class DocumentMetadata(TypedDict, total=False):
    """Basic metadata accepted during document ingestion."""

    title: str
    document_type: str
    version: str


class PDFExtractor(Protocol):
    """Contract for PDF text extraction services."""

    def extract_text_by_page(self, file_path: str) -> list[PDFPage]:
        """Extract structured text content grouped by page."""


class IngestionError(Exception):
    """Raised when PDF ingestion fails in any pipeline step."""


class IngestionService:
    """Orchestrates PDF ingestion into document and chunk persistence layers."""

    def __init__(
        self,
        db: Session,
        *,
        pdf_service: PDFExtractor | None = None,
        chunk_service: ChunkService | None = None,
        embedding_service: EmbeddingService | None = None,
        document_repository: DocumentRepository | None = None,
        chunk_repository: ChunkRepository | None = None,
    ):
        self.db = db
        self.pdf_service = pdf_service or PDFService()
        self.chunk_service = chunk_service or ChunkService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.document_repository = document_repository or DocumentRepository(db)
        self.chunk_repository = chunk_repository or ChunkRepository(db)
        self.logger = logging.getLogger(__name__)

    def ingest_pdf(
        self,
        *,
        file_path: str,
        metadata: DocumentMetadata,
        chunk_size: int = 800,
        chunk_overlap: int | None = None,
    ) -> IngestionResult:
        """Run extraction, chunking, and persistence for one PDF file.

        Args:
            file_path: Path to the source PDF file.
            metadata: Basic document metadata (`title`, `document_type`, `version`).
            chunk_size: Chunk length in characters.
            chunk_overlap: Number of overlapping characters between chunks.

        Returns:
            Structured ingestion result with document and chunk counters.

        Raises:
            IngestionError: If any extraction, chunking, or persistence step fails.
        """
        title = metadata.get("title") or Path(file_path).stem
        document_type = metadata.get("document_type", "unknown")
        version = metadata.get("version", "1.0")

        created_document_id: int | None = None

        try:
            self.logger.info(
                "Ingestion started",
                extra={"file_path": file_path, "title": title, "document_type": document_type, "version": version},
            )

            self.logger.info("Extracting text by page", extra={"file_path": file_path})
            pages = self.pdf_service.extract_text_by_page(file_path)
            self.logger.info("Text extraction completed", extra={"total_pages": len(pages), "file_path": file_path})

            self.logger.info("Creating document record", extra={"file_path": file_path, "title": title})
            document = self.document_repository.create(
                title=title,
                file_path=file_path,
                document_type=document_type,
                version=version,
            )
            created_document_id = document.id
            self.logger.info("Document record created", extra={"document_id": document.id})

            self.logger.info(
                "Generating chunks",
                extra={"document_id": document.id, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
            )
            chunks = self.chunk_service.create_chunks(
                pages,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            self.logger.info("Chunk generation completed", extra={"document_id": document.id, "total_chunks": len(chunks)})

            self.logger.info("Generating embeddings for chunks", extra={"document_id": document.id})
            chunk_payload: list[ChunkCreateInput] = []
            for chunk in chunks:
                chunk_embedding = self.embedding_service.embed_text(chunk["content"])
                chunk_payload.append(
                    {
                        "page_number": chunk["page_number"],
                        "chunk_index": chunk["chunk_index"],
                        "content": chunk["content"],
                        "embedding": chunk_embedding,
                    }
                )
            self.logger.info(
                "Embedding generation completed",
                extra={"document_id": document.id, "embedded_chunks": len(chunk_payload)},
            )
            if chunk_payload:
                self.logger.info(
                    "Saving chunks in database",
                    extra={"document_id": document.id, "chunks_to_save": len(chunk_payload)},
                )
                self.chunk_repository.create_many(
                    document_id=document.id,
                    chunks=chunk_payload,
                )
                self.logger.info("Chunks persisted", extra={"document_id": document.id, "total_chunks": len(chunk_payload)})
            else:
                self.logger.info("No chunks to persist", extra={"document_id": document.id})

            self.logger.info(
                "Ingestion completed successfully",
                extra={"document_id": document.id, "total_pages": len(pages), "total_chunks": len(chunk_payload)},
            )
            return {
                "document_id": document.id,
                "total_pages": len(pages),
                "total_chunks": len(chunk_payload),
            }
        except Exception as exc:
            self.logger.exception(
                "Ingestion failed",
                extra={"file_path": file_path, "created_document_id": created_document_id},
            )
            if created_document_id is not None:
                persisted = self.document_repository.get_by_id(created_document_id)
                if persisted is not None:
                    self.logger.info("Rolling back created document after failure", extra={"document_id": created_document_id})
                    self.db.delete(persisted)
                    self.db.commit()
                    self.logger.info("Rollback completed", extra={"document_id": created_document_id})
            raise IngestionError("Failed to ingest PDF document.") from exc
