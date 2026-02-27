from fastapi import FastAPI, Depends, HTTPException, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import timedelta
import uuid

from app.core.settings import settings
from app.core.security import create_access_token, verify_token, verify_password
from app.core.rate_limit import rate_limiter
from app.core.logging import get_logger
from app.core.graph import LegalRAGWorkflow
from app.api.auth import router as auth_router
from app.llm.gemini_client import list_available_models

logger = get_logger(__name__)

# Security schemes
security = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# Models
class AnalyzeRequest(BaseModel):
    question: str
    documents: Optional[list] = None

# FastAPI instance
app = FastAPI(
    title="Legal RAG Multi-Agent API",
    description="Multi-domain legal analysis system powered by AI agents",
    version="1.0.0"
)

# ========== ADICIONAR CORS AQUI ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-portfolio-beta-cyan.vercel.app",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =========================================

# Include routers
app.include_router(auth_router, prefix="/api/v1", tags=["Authentication"])

# ========== ADICIONAR ENDPOINT OPTIONS PARA DEBUG ==========
@app.options("/api/v1/login")
async def options_login():
    """Handle preflight OPTIONS request"""
    return {"message": "OK"}

@app.options("/api/v1/analyze")
async def options_analyze():
    """Handle preflight OPTIONS request"""
    return {"message": "OK"}
# ============================================================

workflow = LegalRAGWorkflow()

# Dependencies
def verify_api_key(x_api_key: str = Depends(api_key_header)):
    """Verifica API key"""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key missing"
        )
    return x_api_key

def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verifica token JWT"""
    token = credentials.credentials
    username = verify_token(token)
    
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    return username

# Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version
    }

@app.get("/api/v1/models")
async def get_available_models():
    """Retorna modelos Gemini disponíveis"""
    models = list_available_models()
    return {
        "models": models,
        "current_model": "gemini-1.5-flash",
        "total": len(models)
    }

@app.post("/api/v1/analyze")
async def analyze(
    request: AnalyzeRequest,
    x_api_key: str = Depends(verify_api_key),
    username: str = Depends(verify_bearer_token)
):
    """Analisa questão jurídica"""
    try:
        request_id = str(uuid.uuid4())
        
        # Rate limiting
        if not rate_limiter.is_allowed(username):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        documents = request.documents or []
        result = workflow.run(
            question=request.question,
            documents=documents,
            request_id=request_id
        )
        
        logger.info(
            "analysis_completed",
            extra={
                "extra": {
                    "request_id": request_id,
                    "username": username,
                    "risk_level": result.get("risk_level"),
                    "domain": result.get("domain")
                }
            }
        )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "openapi": "/openapi.json",
        "status": "✅ Sistema pronto para análises jurídicas"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)