from unittest.mock import Mock

import pytest

from app.services.embedding_service import (
    EmbeddingProviderAPIError,
    EmbeddingService,
    EmbeddingServiceError,
)


def test_embed_text_returns_numeric_vector_from_provider():
    provider = Mock()
    provider.generate_embedding.return_value = [1, 2.5, 3, 4, 5, 6, 7, 8]

    service = EmbeddingService(provider=provider)
    result = service.embed_text("texto técnico")

    assert result == [1.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    provider.generate_embedding.assert_called_once_with("texto técnico")


def test_embed_text_rejects_empty_input():
    service = EmbeddingService(provider=Mock())

    with pytest.raises(ValueError):
        service.embed_text("   ")


def test_embed_text_wraps_api_errors():
    provider = Mock()
    provider.generate_embedding.side_effect = EmbeddingProviderAPIError("timeout")

    service = EmbeddingService(provider=provider)

    with pytest.raises(EmbeddingServiceError, match="API request failed"):
        service.embed_text("safety rule")


def test_embed_text_rejects_non_numeric_vectors():
    provider = Mock()
    provider.generate_embedding.return_value = [0.1, 0.2, "invalid", 0.4, 0.5, 0.6, 0.7, 0.8]

    service = EmbeddingService(provider=provider)

    with pytest.raises(EmbeddingServiceError, match="contain only numbers"):
        service.embed_text("norma")


def test_embed_text_rejects_invalid_dimensions():
    provider = Mock()
    provider.generate_embedding.return_value = [0.1, 0.2, 0.3]

    service = EmbeddingService(provider=provider)

    with pytest.raises(EmbeddingServiceError, match="invalid dimension"):
        service.embed_text("norma")
