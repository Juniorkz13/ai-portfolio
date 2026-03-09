from collections.abc import Generator
from datetime import datetime
from typing import Protocol

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

router = APIRouter()


class DocumentResponse(BaseModel):
    """Response schema for one indexed document in the catalog."""

    id: int
    title: str
    document_type: str
    version: str
    uploaded_at: datetime
    file_path: str
    filename: str
    total_chunks: int


class DocumentServiceContract(Protocol):
    """Contract for document listing use-cases used by API routes."""

    def list_documents(self) -> list[dict]:
        """List all documents from the catalog."""

    def get_document(self, document_id: int) -> dict | None:
        """Retrieve one document by id."""

    def delete_document(self, document_id: int, *, remove_file: bool = True) -> dict | None:
        """Delete one document by id."""


class DeleteDocumentResponse(BaseModel):
    """Response schema for document deletion endpoint."""

    message: str
    document_id: int
    title: str
    file_path: str
    file_removed: bool


def get_document_service() -> Generator[DocumentServiceContract, None, None]:
    """Build and yield a document service wired with database dependencies."""
    from app.core.database import get_db
    from app.repositories.document_repository import DocumentRepository
    from app.services.document_service import DocumentService

    db_generator = get_db()
    db_session = next(db_generator)
    try:
        repository = DocumentRepository(db_session)
        yield DocumentService(repository)
    finally:
        try:
            next(db_generator)
        except StopIteration:
            pass


@router.get("/documents", response_model=list[DocumentResponse])
def list_documents(
    document_service: DocumentServiceContract = Depends(get_document_service),
) -> list[DocumentResponse]:
    """Return all uploaded/indexed documents available for chat filters."""
    return [DocumentResponse(**item) for item in document_service.list_documents()]


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: int,
    document_service: DocumentServiceContract = Depends(get_document_service),
) -> DocumentResponse:
    """Return one uploaded/indexed document by id."""
    result = document_service.get_document(document_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    return DocumentResponse(**result)


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(
    document_id: int,
    remove_file: bool = Query(default=True),
    document_service: DocumentServiceContract = Depends(get_document_service),
) -> DeleteDocumentResponse:
    """Delete one document and its associated chunks."""
    result = document_service.delete_document(document_id, remove_file=remove_file)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    return DeleteDocumentResponse(
        message="Document deleted successfully.",
        document_id=result["id"],
        title=result["title"],
        file_path=result["file_path"],
        file_removed=result["file_removed"],
    )
