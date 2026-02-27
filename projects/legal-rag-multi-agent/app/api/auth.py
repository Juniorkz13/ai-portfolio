from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from hmac import compare_digest

from app.core.security import create_access_token
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

class LoginRequest(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# DEV ONLY
FAKE_USERS_DB = {
    "admin@example.com": "admin123",
    "user@example.com": "password123",
    "ratelimit_user": "admin",
}

@router.post("/login", response_model=Token)
async def login(credentials: LoginRequest):
    expected_password = FAKE_USERS_DB.get(credentials.username)

    if not expected_password or not compare_digest(credentials.password, expected_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": credentials.username})
    return {"access_token": access_token, "token_type": "bearer"}