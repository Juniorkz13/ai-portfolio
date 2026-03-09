from fastapi import FastAPI

from app.api.routes_chat import router as chat_router
from app.api.routes_upload import router as upload_router


app = FastAPI(
    title="AI Technical Norms Assistant API",
    version="0.1.0",
    description="Initial FastAPI backend scaffold for the RAG architecture project.",
)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(upload_router, prefix="/api/upload", tags=["upload"])
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
