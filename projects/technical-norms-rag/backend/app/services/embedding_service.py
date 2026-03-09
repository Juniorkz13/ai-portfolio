import logging
from typing import Protocol

from app.core.config import settings


class EmbeddingProviderAPIError(Exception):
    """Raised by providers when an external embedding API request fails."""


class EmbeddingServiceError(Exception):
    """Raised by `EmbeddingService` when embedding generation fails."""


class EmbeddingProvider(Protocol):
    """Provider contract to allow easy swapping of embedding backends."""

    def generate_embedding(self, text: str) -> list[float]:
        """Generate one embedding vector for input text."""


class HashEmbeddingProvider:
    """Deterministic local fallback provider used for development/testing."""

    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def generate_embedding(self, text: str) -> list[float]:
        base = abs(hash(text))
        # Fixed-size lightweight vector to keep service usable without API calls.
        return [float((base >> (idx * 8)) & 0xFF) / 255.0 for idx in range(self.dimensions)]


class EmbeddingService:
    """Service layer that validates input and delegates embedding generation."""

    def __init__(self, provider: EmbeddingProvider | None = None):
        self.embedding_dimension = settings.embedding_dimension
        self.provider = provider or HashEmbeddingProvider(self.embedding_dimension)
        self.logger = logging.getLogger(__name__)

    def embed_text(self, text: str) -> list[float]:
        """Generate a numeric embedding vector for a single text string.

        Args:
            text: Raw text content to be embedded.

        Returns:
            A numeric embedding vector.

        Raises:
            ValueError: If input text is empty.
            EmbeddingServiceError: If provider/API fails or returns invalid output.
        """
        self.logger.info("Embedding request received", extra={"text_length": len(text or "")})
        if not text or not text.strip():
            raise ValueError("text must not be empty.")

        try:
            self.logger.info("Generating embedding vector")
            vector = self.provider.generate_embedding(text)
            self.logger.info("Embedding vector generated", extra={"embedding_dimensions": len(vector) if isinstance(vector, list) else 0})
        except EmbeddingProviderAPIError as exc:
            self.logger.exception("Embedding provider API request failed")
            raise EmbeddingServiceError("Embedding API request failed.") from exc
        except Exception as exc:
            self.logger.exception("Unexpected embedding provider error")
            raise EmbeddingServiceError("Unexpected embedding provider error.") from exc

        if not isinstance(vector, list) or not vector:
            raise EmbeddingServiceError("Provider returned an invalid embedding vector.")

        if not all(isinstance(item, (int, float)) for item in vector):
            raise EmbeddingServiceError("Embedding vector must contain only numbers.")
        if len(vector) != self.embedding_dimension:
            raise EmbeddingServiceError(
                f"Embedding vector has invalid dimension. Expected {self.embedding_dimension}, got {len(vector)}."
            )

        self.logger.info("Embedding request completed successfully")
        return [float(value) for value in vector]
