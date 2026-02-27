from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from datetime import timedelta

from app.core.security import create_access_token, verify_password
from app.core.settings import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["auth"])

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str

# Demo users
DEMO_USERS = {
    "user@example.com": "password123",
    "admin@example.com": "admin123",
    "ratelimit_user": "admin"
}

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Realiza login e retorna JWT token"""
    try:
        if request.username not in DEMO_USERS:
            raise HTTPException(status_code=401, detail="Credenciais inválidas")
        
        stored_password = DEMO_USERS[request.username]
        if request.password != stored_password:
            raise HTTPException(status_code=401, detail="Credenciais inválidas")
        
        access_token_expires = timedelta(minutes=settings.jwt_expiration_minutes)
        access_token = create_access_token(
            data={"sub": request.username},
            expires_delta=access_token_expires
        )
        
        logger.info(
            "user_login",
            extra={"extra": {"username": request.username}}
        )
        
        return {"access_token": access_token, "token_type": "bearer"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")