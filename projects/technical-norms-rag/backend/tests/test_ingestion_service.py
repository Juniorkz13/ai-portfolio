from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
import pytest

from app.core.database import Base
from app.models.chunk import Chunk
from app.models.document import Document
from app.services.chunk_service import ChunkService
from app.services.ingestion_service import IngestionError, IngestionService


class FakePDFService:
    def __init__(self, pages: list[dict[str, int | str]]):
        self.pages = pages

    def extract_text_by_page(self, file_path: str) -> list[dict[str, int | str]]:
        _ = file_path
        return self.pages


class FailingPDFService:
    def extract_text_by_page(self, file_path: str) -> list[dict[str, int | str]]:
        _ = file_path
        raise RuntimeError("pdf parsing failed")


def _make_session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return Session(bind=engine)


def test_ingest_pdf_success_persists_document_and_chunks():
    session = _make_session()
    service = IngestionService(
        session,
        pdf_service=FakePDFService(
            [
                {"page_number": 1, "text": "Regras de evacuacao e acessibilidade."},
                {"page_number": 2, "text": "   "},
            ]
        ),
        chunk_service=ChunkService(),
    )

    result = service.ingest_pdf(
        file_path="storage/pdfs/norma.pdf",
        metadata={"title": "Norma", "document_type": "regulation", "version": "2026"},
        chunk_size=30,
        chunk_overlap=5,
    )

    persisted_document = session.get(Document, result["document_id"])
    persisted_chunks = session.scalars(
        select(Chunk).where(Chunk.document_id == result["document_id"])
    ).all()

    assert persisted_document is not None
    assert result["total_pages"] == 2
    assert result["total_chunks"] == len(persisted_chunks)
    assert len(persisted_chunks) > 0
    assert all(chunk.document_id == result["document_id"] for chunk in persisted_chunks)

    session.close()


def test_ingest_pdf_raises_ingestion_error_on_extraction_failure():
    session = _make_session()
    service = IngestionService(
        session,
        pdf_service=FailingPDFService(),
        chunk_service=ChunkService(),
    )

    with pytest.raises(IngestionError):
        service.ingest_pdf(
            file_path="storage/pdfs/invalid.pdf",
            metadata={"title": "Invalid"},
        )

    assert session.scalars(select(Document)).all() == []
    assert session.scalars(select(Chunk)).all() == []

    session.close()
