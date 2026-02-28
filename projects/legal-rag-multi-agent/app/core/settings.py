from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Configurações centralizadas da aplicação"""
    
    # Gemini API
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")
    
    # JWT
    jwt_secret_key: str = Field(
        default="your-secret-key-change-in-production",
        env="JWT_SECRET_KEY"
    )
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_minutes: int = Field(default=30, env="JWT_EXPIRATION_MINUTES")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(default=3600, env="RATE_LIMIT_WINDOW_SECONDS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # App
    app_name: str = Field(default="Legal RAG Multi-Agent", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()