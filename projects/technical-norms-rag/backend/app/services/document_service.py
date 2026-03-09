from datetime import datetime
import logging
from pathlib import Path
from typing import TypedDict

from app.models.document import Document
from app.repositories.document_repository import DocumentRepository


class DocumentListItem(TypedDict):
    """Serialized document item returned by listing endpoints."""

    id: int
    title: str
    document_type: str
    version: str
    uploaded_at: datetime
    file_path: str
    filename: str
    total_chunks: int


class DocumentService:
    """Use-case layer for document listing and lookup."""

    def __init__(self, repository: DocumentRepository):
        self.repository = repository
        self.logger = logging.getLogger(__name__)

    def list_documents(self) -> list[DocumentListItem]:
        """Return all documents serialized for API responses."""
        documents = self.repository.list_with_chunks()
        return [self._serialize_document(document) for document in documents]

    def get_document(self, document_id: int) -> DocumentListItem | None:
        """Return one serialized document by id, or `None` if missing."""
        document = self.repository.get_by_id_with_chunks(document_id)
        if document is None:
            return None
        return self._serialize_document(document)

    def delete_document(self, document_id: int, *, remove_file: bool = True) -> dict | None:
        """Delete one document and optionally delete its physical PDF file."""
        document = self.repository.delete_by_id(document_id)
        if document is None:
            return None

        file_removed = False
        if remove_file:
            file_path = Path(document.file_path)
            try:
                if file_path.exists():
                    file_path.unlink()
                    file_removed = True
            except Exception:
                # Keep API stable even when file cleanup fails.
                self.logger.exception(
                    "Failed to remove document file after DB deletion",
                    extra={"document_id": document_id, "file_path": str(file_path)},
                )

        return {
            "id": document.id,
            "title": document.title,
            "file_path": document.file_path,
            "file_removed": file_removed,
        }

    def _serialize_document(self, document: Document) -> DocumentListItem:
        """Map ORM entity into API-safe dictionary payload."""
        return {
            "id": document.id,
            "title": document.title,
            "document_type": document.document_type,
            "version": document.version,
            "uploaded_at": document.uploaded_at,
            "file_path": document.file_path,
            "filename": Path(document.file_path).name,
            "total_chunks": len(document.chunks),
        }
