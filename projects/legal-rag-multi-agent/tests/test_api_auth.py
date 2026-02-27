from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_analyze_requires_api_key():
    """Testa se /analyze retorna 401 sem API key"""
    response = client.post(
        "/api/v1/analyze",
        json={"question": "test question"}
    )
    assert response.status_code == 401

def test_analyze_with_valid_api_key():
    """Testa /analyze com API key e token válidos"""
    # Fazer login
    login = client.post(
        "/api/v1/login",
        json={"username": "user@example.com", "password": "password123"}
    )
    assert login.status_code == 200
    token = login.json()["access_token"]
    
    # Fazer requisição com token
    response = client.post(
        "/api/v1/analyze",
        json={"question": "What is labor law?"},
        headers={
            "X-API-Key": "test-key",
            "Authorization": f"Bearer {token}"
        }
    )
    assert response.status_code == 200
    assert "risk_level" in response.json()