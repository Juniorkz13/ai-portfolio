from collections.abc import Generator
import logging
from typing import Protocol

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Input payload for question answering endpoint."""

    question: str
    top_k: int = Field(default=5, gt=0, le=20)
    document_id: int | None = Field(default=None, gt=0)
    document_type: str | None = None
    version: str | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Ensure question contains meaningful non-whitespace text."""
        if not value or not value.strip():
            raise ValueError("question must not be empty.")
        return value.strip()


class SourceResponse(BaseModel):
    """Source reference returned in chat response."""

    document_id: int
    document_title: str
    document_type: str
    version: str
    page_number: int
    chunk_index: int
    excerpt: str


class ChatAPIResponse(BaseModel):
    """Response payload for grounded chat answers."""

    answer: str
    explanation: str
    sources: list[SourceResponse]


class ChatRunner(Protocol):
    """Contract for chat service dependency used by the route."""

    def answer(
        self,
        question: str,
        top_k: int = 5,
        filters: dict[str, int | str] | None = None,
    ) -> ChatAPIResponse | dict:
        """Process user question and return structured answer."""


def get_chat_service() -> Generator[ChatRunner, None, None]:
    """Build and yield chat service using configured dependencies."""
    from app.core.database import get_db
    from app.services.chat_service import ChatService
    from app.services.llm.gemini_client import GeminiClient
    from app.services.retrieval_service import RetrievalService

    db_generator = get_db()
    db_session = next(db_generator)
    try:
        retrieval_service = RetrievalService(db=db_session)
        gemini_client = GeminiClient()
        yield ChatService(
            retrieval_service=retrieval_service,
            llm_client=gemini_client,
        )
    except Exception:
        logger.exception("Failed to initialize chat service dependencies")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat request.",
        )
    finally:
        try:
            next(db_generator)
        except StopIteration:
            pass


@router.post("/chat", response_model=ChatAPIResponse)
def ask_question(
    payload: ChatRequest,
    chat_service: ChatRunner = Depends(get_chat_service),
) -> ChatAPIResponse:
    """Receive a natural language question and return grounded answer fields."""
    from app.services.chat_service import ChatServiceError

    filters: dict[str, int | str] = {}
    if payload.document_id is not None:
        filters["document_id"] = payload.document_id
    if payload.document_type:
        filters["document_type"] = payload.document_type
    if payload.version:
        filters["version"] = payload.version

    try:
        result = chat_service.answer(
            payload.question,
            top_k=payload.top_k,
            filters=filters or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ChatServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat request.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while processing chat request.",
        ) from exc

    return ChatAPIResponse(**result)
