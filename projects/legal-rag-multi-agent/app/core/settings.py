from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Configurações centralizadas da aplicação"""
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="allow",
        env_file_encoding="utf-8"
    )
    
    # Gemini API
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    
    # JWT
    jwt_secret_key: str = Field(
        default="your-secret-key-change-in-production",
        validation_alias="JWT_SECRET_KEY"
    )
    jwt_algorithm: str = Field(default="HS256", validation_alias="JWT_ALGORITHM")
    jwt_expiration_minutes: int = Field(default=30, validation_alias="JWT_EXPIRATION_MINUTES")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, validation_alias="RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(default=3600, validation_alias="RATE_LIMIT_WINDOW_SECONDS")
    
    # Logging
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    
    # App
    app_name: str = Field(default="Legal RAG Multi-Agent", validation_alias="APP_NAME")
    app_version: str = Field(default="1.0.0", validation_alias="APP_VERSION")
    debug: bool = Field(default=False, validation_alias="DEBUG")

def get_settings() -> Settings:
    """Retorna instância de Settings"""
    return Settings()

settings = get_settings()