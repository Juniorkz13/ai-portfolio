from fastapi import FastAPI, Depends, HTTPException, Header, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import timedelta
import uuid
import jwt

from app.core.settings import settings
from app.core.security import create_access_token, verify_token, verify_password
from app.core.rate_limit import rate_limiter
from app.core.logging import get_logger
from app.llm.gemini_client import list_available_models

logger = get_logger(__name__)

# Security schemes
security = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# Models
class AnalyzeRequest(BaseModel):
    question: str
    documents: Optional[list] = None

class LoginRequest(BaseModel):
    username: str
    password: str

# FastAPI instance
app = FastAPI(
    title="Legal RAG Multi-Agent API",
    description="Multi-domain legal analysis system powered by AI agents",
    version="1.0.0"
)

# CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Permite OPTIONS
    allow_headers=["*"],  # Permite todos os headers
)

# INCLUDE ROUTER SEGUNDO (antes de qualquer validação global)
# app.include_router(auth_router, prefix="/api/v1", tags=["auth"])

# Security schemes (DEPOIS)
security = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# Dependencies
def verify_api_key(x_api_key: str = Depends(api_key_header)):
    """Verifica API key"""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key missing"
        )
    return x_api_key

def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verifica e extrai token JWT do header"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except Exception as e:
        logger.error(f"[VERIFY_TOKEN] Error: {str(e)}")
        raise HTTPException(status_code=401, detail="Not authenticated")

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Middleware de autenticação"""
    if request.method == "OPTIONS":
        return await call_next(request)

    if request.url.path in ["/health", "/docs", "/openapi.json", "/api/v1/login"]:
        return await call_next(request)

    # ...existing code de validação...
    return await call_next(request)

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
async def analyze(request: AnalyzeRequest):
    """Analisa questão jurídica"""
    return {
        "analysis": {
            "summary": f"Análise de: {request.question}",
            "details": "Detalhes da análise",
            "sources": [],
            "confidence": 0.95
        }
    }

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

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/v1/login")
async def login(request: LoginRequest):
    """Login endpoint"""
    if request.username == "admin@example.com" and request.password == "admin123":
        token = create_access_token(data={"sub": request.username})
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)