from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import answer_question
from app.bootstrap import init_rag
from app.config import get_settings
from app import state
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi.responses import JSONResponse
import uuid

app = FastAPI(
    title="RAG Enterprise Assistant",
    version="1.2.0"
)

limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Try again later."}
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
@limiter.limit("5/minute")
def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    answer = answer_question(request.question, session_id)
    return {
        "answer": answer,
        "session_id": session_id
    }

@app.get("/health")
def health():
    settings = get_settings()

    return {
        "status": "ok",
        "env": settings["env"],
        "rag": {
            "ready": state.faiss_index is not None and state.chunks is not None,
            "chunks": len(state.chunks) if state.chunks else 0
        },
        "llm": {
            "provider": "gemini",
            "model": settings["gemini_model"]
        }
    }