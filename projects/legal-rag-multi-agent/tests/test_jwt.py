from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

def test_login_and_analyze():
    """Testa fluxo de login e análise"""
    # Fazer login
    login = client.post(
        "/api/v1/login",
        json={"username": "user@example.com", "password": "password123"}
    )
    assert login.status_code == 200
    
    data = login.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    
    token = data["access_token"]
    
    # Usar token para fazer análise
    response = client.post(
        "/api/v1/analyze",
        json={"question": "What about employment rights?"},
        headers={
            "X-API-Key": "test-key",
            "Authorization": f"Bearer {token}"
        }
    )
    assert response.status_code == 200
    result = response.json()
    assert "risk_level" in result
    assert "analysis" in result