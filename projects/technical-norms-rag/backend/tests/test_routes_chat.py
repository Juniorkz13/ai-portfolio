from fastapi.testclient import TestClient

from app.api.routes_chat import get_chat_service
from app.main import app
from app.services.chat_service import ChatServiceError


class FakeChatService:
    def answer(
        self,
        question: str,
        top_k: int = 5,
        filters: dict[str, int | str] | None = None,
    ) -> dict:
        _ = (question, top_k, filters)
        return {
            "answer": "A largura mínima é 1,20 m.",
            "explanation": "Baseado no trecho técnico recuperado da norma.",
            "sources": [
                {
                    "document_id": 7,
                    "document_title": "IT-01",
                    "document_type": "fire_regulation",
                    "version": "2026",
                    "page_number": 14,
                    "chunk_index": 2,
                    "excerpt": "Trecho curto da norma recuperada.",
                },
            ],
        }


class FailingChatService:
    def answer(
        self,
        question: str,
        top_k: int = 5,
        filters: dict[str, int | str] | None = None,
    ) -> dict:
        _ = (question, top_k, filters)
        raise ChatServiceError("failure")


def test_chat_valid_question_returns_answer_structure():
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    client = TestClient(app)

    response = client.post("/chat", json={"question": "Qual a largura mínima da escada?", "top_k": 3})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "A largura mínima é 1,20 m."
    assert "trecho técnico" in payload["explanation"]
    assert payload["sources"] == [
        {
            "document_id": 7,
            "document_title": "IT-01",
            "document_type": "fire_regulation",
            "version": "2026",
            "page_number": 14,
            "chunk_index": 2,
            "excerpt": "Trecho curto da norma recuperada.",
        }
    ]

    app.dependency_overrides.clear()


def test_chat_empty_question_returns_validation_error():
    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()
    client = TestClient(app)

    response = client.post("/chat", json={"question": "   "})

    assert response.status_code == 422

    app.dependency_overrides.clear()


def test_chat_service_failure_returns_500():
    app.dependency_overrides[get_chat_service] = lambda: FailingChatService()
    client = TestClient(app)

    response = client.post("/chat", json={"question": "Pergunta válida"})

    assert response.status_code == 500
    assert "Failed to process chat request" in response.json()["detail"]

    app.dependency_overrides.clear()


def test_chat_forwards_optional_filters():
    captured: dict[str, object] = {}

    class CaptureChatService(FakeChatService):
        def answer(
            self,
            question: str,
            top_k: int = 5,
            filters: dict[str, int | str] | None = None,
        ) -> dict:
            captured["question"] = question
            captured["top_k"] = top_k
            captured["filters"] = filters
            return super().answer(question, top_k=top_k, filters=filters)

    app.dependency_overrides[get_chat_service] = lambda: CaptureChatService()
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "question": "Qual regra se aplica?",
            "top_k": 4,
            "document_id": 7,
            "document_type": "fire_regulation",
            "version": "2026",
        },
    )

    assert response.status_code == 200
    assert captured["top_k"] == 4
    assert captured["filters"] == {
        "document_id": 7,
        "document_type": "fire_regulation",
        "version": "2026",
    }

    app.dependency_overrides.clear()
