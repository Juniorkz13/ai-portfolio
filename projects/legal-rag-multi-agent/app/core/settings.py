from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Configurações centralizadas da aplicação"""
    
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")
    jwt_secret_key: str = Field(default="your-secret-key-change-in-production", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_minutes: int = Field(default=30, env="JWT_EXPIRATION_MINUTES")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(default=3600, env="RATE_LIMIT_WINDOW_SECONDS")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    app_name: str = Field(default="Legal RAG Multi-Agent", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    class Config:
        env_file = ".env"

settings = Settings()