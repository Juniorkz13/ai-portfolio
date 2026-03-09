from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables or .env file.
    """

    # =========================
    # Environment
    # =========================
    environment: str = "development"
    debug: bool = True

    # =========================
    # Database
    # =========================
    database_url: str = "postgresql+psycopg://rag_user:rag_password@localhost:5432/rag_architecture_ai"

    # =========================
    # Storage
    # =========================
    upload_dir: str = "storage/pdfs"

    # =========================
    # Embeddings / LLM
    # =========================
    llm_provider: str = "gemini"
    embedding_provider: str = "gemini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-flash-latest"
    embedding_dimension: int = 8

    # =========================
    # RAG configuration
    # =========================
    chunk_size: int = 800
    chunk_overlap: int = 150
    retrieval_top_k: int = 5

    # =========================
    # Pydantic settings config
    # =========================
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, value: str) -> str:
        """Enforce the only supported LLM provider for this project."""
        normalized = value.strip().lower()
        if normalized != "gemini":
            raise ValueError("Only 'gemini' is supported as LLM provider.")
        return normalized

    @field_validator("gemini_model")
    @classmethod
    def validate_gemini_model(cls, value: str) -> str:
        """Enforce the only allowed Gemini model name."""
        model_name = value.strip()
        if model_name != "gemini-flash-latest":
            raise ValueError("Only 'gemini-flash-latest' is allowed as Gemini model.")
        return model_name


# Singleton settings instance
settings = Settings()


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_PATH = BASE_DIR / settings.upload_dir

# garante que a pasta de upload exista
UPLOAD_PATH.mkdir(parents=True, exist_ok=True)
