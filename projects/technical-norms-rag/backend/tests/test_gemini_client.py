import sys
import types

import pytest

from app.core.config import settings
from app.services.llm.gemini_client import (
    GeminiClient,
    GeminiConfigurationError,
    GeminiGenerationError,
)


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_content(self, prompt: str):
        _ = prompt
        return _FakeResponse("generated text")


class _FakeGenAI:
    @staticmethod
    def configure(api_key: str):
        _ = api_key

    GenerativeModel = _FakeModel


def test_gemini_client_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_model", "gemini-flash-latest")
    monkeypatch.setattr(settings, "gemini_api_key", None)

    with pytest.raises(GeminiConfigurationError, match="GEMINI_API_KEY"):
        GeminiClient()


def test_gemini_client_enforces_allowed_model(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_model", "gemini-2.0-flash")
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")

    with pytest.raises(GeminiConfigurationError, match="gemini-flash-latest"):
        GeminiClient()


def test_gemini_client_generate_uses_fixed_model(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_model", "gemini-flash-latest")
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")

    fake_google = types.SimpleNamespace(generativeai=_FakeGenAI)
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.generativeai", _FakeGenAI)

    client = GeminiClient()
    output = client.generate("prompt", model="ignored-model")

    assert output == "generated text"


def test_gemini_client_generate_raises_on_empty_response(monkeypatch):
    class _EmptyModel:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def generate_content(self, prompt: str):
            _ = prompt
            return types.SimpleNamespace(text="")

    class _EmptyGenAI:
        @staticmethod
        def configure(api_key: str):
            _ = api_key

        GenerativeModel = _EmptyModel

    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "gemini_model", "gemini-flash-latest")
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")

    fake_google = types.SimpleNamespace(generativeai=_EmptyGenAI)
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.generativeai", _EmptyGenAI)

    client = GeminiClient()

    with pytest.raises(GeminiGenerationError):
        client.generate("prompt")
