from collections.abc import Generator
import logging
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


class UploadResponse(BaseModel):
    """Structured response returned after successful PDF ingestion."""

    message: str
    document_id: int
    total_pages: int
    total_chunks: int


class IngestionResult(BaseModel):
    """Ingestion result contract used by the upload route."""

    document_id: int
    total_pages: int
    total_chunks: int


class IngestionRunner(Protocol):
    """Dependency contract for PDF ingestion orchestration."""

    def ingest_pdf(
        self,
        *,
        file_path: str,
        metadata: dict[str, str],
        chunk_size: int = 800,
        chunk_overlap: int | None = None,
    ) -> IngestionResult | dict[str, int]:
        """Run ingestion pipeline for one PDF path and return counters."""


def get_ingestion_service() -> Generator[IngestionRunner, None, None]:
    """Build and yield an ingestion service using the configured DB session."""
    from app.core.database import get_db
    from app.services.ingestion_service import IngestionService

    db_generator = get_db()
    db_session = next(db_generator)
    try:
        yield IngestionService(db_session)
    finally:
        try:
            next(db_generator)
        except StopIteration:
            pass


def _validate_pdf_upload(file: UploadFile) -> None:
    """Validate that upload has PDF extension and compatible content type."""
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed.",
        )

    if file.content_type and file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid content type. Expected application/pdf.",
        )


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    document_type: str = Form(default="unknown"),
    version: str = Form(default="1.0"),
    ingestion_service: IngestionRunner = Depends(get_ingestion_service),
) -> UploadResponse:
    """Upload one PDF, persist it in storage, and trigger ingestion pipeline."""
    logger.info(
        "Upload request received",
        extra={
            "uploaded_filename": file.filename,
            "content_type": file.content_type,
            "document_type": document_type,
            "version": version,
        },
    )
    _validate_pdf_upload(file)

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{uuid4().hex}_{Path(file.filename or 'document.pdf').name}"
    destination_path = upload_dir / safe_name

    try:
        logger.info("Saving uploaded PDF file", extra={"destination_path": str(destination_path)})
        content = await file.read()
        destination_path.write_bytes(content)
        logger.info(
            "Uploaded PDF file saved",
            extra={"destination_path": str(destination_path), "size_bytes": len(content)},
        )
    except Exception as exc:
        logger.exception(
            "Failed to persist uploaded PDF file",
            extra={"destination_path": str(destination_path), "uploaded_filename": file.filename},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist uploaded PDF file.",
        ) from exc
    finally:
        await file.close()

    try:
        logger.info("Starting ingestion pipeline", extra={"file_path": str(destination_path)})
        result = ingestion_service.ingest_pdf(
            file_path=str(destination_path),
            metadata={
                "title": title or Path(file.filename or "document.pdf").stem,
                "document_type": document_type,
                "version": version,
            },
        )
        logger.info(
            "Ingestion pipeline completed",
            extra={
                "document_id": result["document_id"],
                "total_pages": result["total_pages"],
                "total_chunks": result["total_chunks"],
            },
        )
    except Exception as exc:
        logger.exception(
            "Failed to process uploaded PDF",
            extra={"file_path": str(destination_path), "uploaded_filename": file.filename},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process uploaded PDF.",
        ) from exc

    return UploadResponse(
        message="PDF uploaded and processed successfully.",
        document_id=result["document_id"],
        total_pages=result["total_pages"],
        total_chunks=result["total_chunks"],
    )
