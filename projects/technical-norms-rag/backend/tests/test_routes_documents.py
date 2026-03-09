from datetime import datetime

from fastapi.testclient import TestClient

from app.api.routes_documents import get_document_service
from app.main import app


class FakeDocumentService:
    def __init__(self):
        self.last_remove_file: bool | None = None

    def list_documents(self) -> list[dict]:
        return [
            {
                "id": 1,
                "title": "IT-01",
                "document_type": "fire_regulation",
                "version": "2026",
                "uploaded_at": datetime(2026, 3, 1, 10, 0, 0),
                "file_path": "storage/pdfs/it-01.pdf",
                "filename": "it-01.pdf",
                "total_chunks": 12,
            }
        ]

    def get_document(self, document_id: int) -> dict | None:
        if document_id == 1:
            return self.list_documents()[0]
        return None

    def delete_document(self, document_id: int, *, remove_file: bool = True) -> dict | None:
        self.last_remove_file = remove_file
        if document_id != 1:
            return None
        return {
            "id": 1,
            "title": "IT-01",
            "file_path": "storage/pdfs/it-01.pdf",
            "file_removed": remove_file,
        }


def test_list_documents_returns_catalog_items():
    app.dependency_overrides[get_document_service] = lambda: FakeDocumentService()
    client = TestClient(app)

    response = client.get("/documents")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["id"] == 1
    assert payload[0]["title"] == "IT-01"
    assert payload[0]["total_chunks"] == 12

    app.dependency_overrides.clear()


def test_get_document_returns_single_item():
    app.dependency_overrides[get_document_service] = lambda: FakeDocumentService()
    client = TestClient(app)

    response = client.get("/documents/1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == 1
    assert payload["filename"] == "it-01.pdf"

    app.dependency_overrides.clear()


def test_get_document_returns_404_when_missing():
    app.dependency_overrides[get_document_service] = lambda: FakeDocumentService()
    client = TestClient(app)

    response = client.get("/documents/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."

    app.dependency_overrides.clear()


def test_delete_document_returns_success():
    fake_service = FakeDocumentService()
    app.dependency_overrides[get_document_service] = lambda: fake_service
    client = TestClient(app)

    response = client.delete("/documents/1?remove_file=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["message"] == "Document deleted successfully."
    assert payload["document_id"] == 1
    assert payload["file_removed"] is True
    assert fake_service.last_remove_file is True

    app.dependency_overrides.clear()


def test_delete_document_returns_404_when_missing():
    app.dependency_overrides[get_document_service] = lambda: FakeDocumentService()
    client = TestClient(app)

    response = client.delete("/documents/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."

    app.dependency_overrides.clear()


def test_delete_document_can_disable_file_removal():
    fake_service = FakeDocumentService()
    app.dependency_overrides[get_document_service] = lambda: fake_service
    client = TestClient(app)

    response = client.delete("/documents/1?remove_file=false")

    assert response.status_code == 200
    assert fake_service.last_remove_file is False

    app.dependency_overrides.clear()
