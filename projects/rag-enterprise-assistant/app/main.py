from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import answer_question
from app.bootstrap import init_rag
import uuid

app = FastAPI(
    title="RAG Enterprise Assistant",
    version="1.2.0"
)


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str


@app.on_event("startup")
def startup_event():
    init_rag()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    answer = answer_question(request.question, session_id)
    return {
        "answer": answer,
        "session_id": session_id
    }