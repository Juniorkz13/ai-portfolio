import os
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Carrega variáveis de ambiente para testes ANTES de importar a app
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-very-long-and-secure-minimum-32-chars")
os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("JWT_EXPIRATION_MINUTES", "30")
os.environ.setdefault("RATE_LIMIT_REQUESTS", "100")
os.environ.setdefault("RATE_LIMIT_WINDOW_SECONDS", "3600")
os.environ.setdefault("LOG_LEVEL", "INFO")
os.environ.setdefault("APP_NAME", "Legal RAG Multi-Agent")
os.environ.setdefault("APP_VERSION", "1.0.0")
os.environ.setdefault("DEBUG", "false")

import pytest
from fastapi.testclient import TestClient
from app.api.main import app

@pytest.fixture(scope="session")
def client():
    """Fixture para TestClient da FastAPI"""
    return TestClient(app)

@pytest.fixture
def test_token(client):
    """Fixture para gerar token de teste"""
    response = client.post(
        "/api/v1/login",
        json={"username": "user@example.com", "password": "password123"}
    )
    if response.status_code == 200:
        return response.json()["access_token"]
    return None