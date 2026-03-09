from pathlib import Path

from fastapi.testclient import TestClient

from app.api.routes_upload import get_ingestion_service
from app.core.config import settings
from app.main import app


class FakeIngestionService:
    def ingest_pdf(self, *, file_path: str, metadata: dict[str, str], chunk_size: int = 800, chunk_overlap: int | None = None) -> dict[str, int]:
        _ = (file_path, metadata, chunk_size, chunk_overlap)
        return {"document_id": 42, "total_pages": 3, "total_chunks": 9}


class FailingIngestionService:
    def ingest_pdf(self, *, file_path: str, metadata: dict[str, str], chunk_size: int = 800, chunk_overlap: int | None = None) -> dict[str, int]:
        _ = (file_path, metadata, chunk_size, chunk_overlap)
        raise RuntimeError("ingestion failure")


def test_upload_pdf_valid_file(tmp_path):
    original_upload_dir = settings.upload_dir
    settings.upload_dir = str(tmp_path)

    app.dependency_overrides[get_ingestion_service] = lambda: FakeIngestionService()
    client = TestClient(app)

    response = client.post(
        "/upload",
        files={"file": ("norma.pdf", b"%PDF-1.4 mock", "application/pdf")},
        data={"title": "Norma", "document_type": "regulation", "version": "1.0"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["message"] == "PDF uploaded and processed successfully."
    assert payload["document_id"] == 42
    assert payload["total_pages"] == 3
    assert payload["total_chunks"] == 9

    saved_files = list(Path(tmp_path).glob("*.pdf"))
    assert saved_files

    app.dependency_overrides.clear()
    settings.upload_dir = original_upload_dir


def test_upload_pdf_invalid_file_type(tmp_path):
    original_upload_dir = settings.upload_dir
    settings.upload_dir = str(tmp_path)

    app.dependency_overrides[get_ingestion_service] = lambda: FakeIngestionService()
    client = TestClient(app)
    response = client.post(
        "/upload",
        files={"file": ("not-pdf.txt", b"plain text", "text/plain")},
    )

    assert response.status_code == 400
    assert "Only PDF files are allowed" in response.json()["detail"]

    app.dependency_overrides.clear()
    settings.upload_dir = original_upload_dir


def test_upload_pdf_returns_error_when_ingestion_fails(tmp_path):
    original_upload_dir = settings.upload_dir
    settings.upload_dir = str(tmp_path)

    app.dependency_overrides[get_ingestion_service] = lambda: FailingIngestionService()
    client = TestClient(app)

    response = client.post(
        "/upload",
        files={"file": ("norma.pdf", b"%PDF-1.4 mock", "application/pdf")},
    )

    assert response.status_code == 500
    assert "Failed to process uploaded PDF" in response.json()["detail"]

    app.dependency_overrides.clear()
    settings.upload_dir = original_upload_dir
