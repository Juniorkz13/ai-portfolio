from fastapi import FastAPI

from app.api.routes_chat import router as chat_router
from app.api.routes_documents import router as documents_router
from app.api.routes_upload import router as upload_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    application = FastAPI(
        title="AI Technical Norms Assistant API",
        version="0.1.0",
        description="Backend API for PDF ingestion and grounded chat responses.",
    )

    @application.get("/health", tags=["health"])
    def health_check() -> dict[str, str]:
        """Health endpoint used for liveness checks."""
        return {"status": "ok"}

    application.include_router(upload_router, tags=["upload"])
    application.include_router(chat_router, tags=["chat"])
    application.include_router(documents_router, tags=["documents"])

    return application


app = create_app()
