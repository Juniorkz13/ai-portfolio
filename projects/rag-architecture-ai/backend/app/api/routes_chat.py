from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ChatRequest(BaseModel):
    question: str


@router.post("/")
def ask_question(payload: ChatRequest) -> dict[str, str]:
    # Placeholder: run retrieval + generation pipeline and return grounded answer.
    return {
        "answer": "Chat route scaffolded.",
        "question": payload.question,
    }
