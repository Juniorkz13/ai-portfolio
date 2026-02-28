from fastapi import Depends, HTTPException, status, Header
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.core.settings import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/login",   
    auto_error=False
)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifica se a senha está correta"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Gera hash da senha"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Cria token JWT"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expiration_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.jwt_secret_key, 
        algorithm=settings.jwt_algorithm
    )
    
    return encoded_jwt

def verify_token(token: str) -> Optional[str]:
    """Verifica token JWT e retorna username"""
    try:
        payload = jwt.decode(
            token, 
            settings.jwt_secret_key, 
            algorithms=[settings.jwt_algorithm]
        )
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError:
        return None

def verify_bearer_token(authorization: Optional[str] = Header(None)) -> str:
    """Verifica e extrai token JWT do header"""
    logger = logging.getLogger(__name__)
    logger.info(f"[VERIFY_TOKEN] Authorization header: {authorization[:20] if authorization else 'None'}...")
    
    if not authorization:
        logger.error("[VERIFY_TOKEN] No authorization header")
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    if not authorization.startswith("Bearer "):
        logger.error("[VERIFY_TOKEN] Invalid token format")
        raise HTTPException(status_code=401, detail="Invalid token format")
    
    token = authorization.replace("Bearer ", "")
    
    try:
        logger.info("[VERIFY_TOKEN] Decoding token...")
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        username = payload.get("sub")
        logger.info(f"[VERIFY_TOKEN] Token valid for user: {username}")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except Exception as e:
        logger.error(f"[VERIFY_TOKEN] Decode failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Token expired or invalid")