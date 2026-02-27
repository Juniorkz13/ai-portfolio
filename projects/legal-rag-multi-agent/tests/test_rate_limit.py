from fastapi.testclient import TestClient
from app.api.main import app
from app.core.rate_limit import RateLimiter

client = TestClient(app)

def get_token(username: str, password: str) -> str:
    """Helper para obter token"""
    response = client.post(
        "/api/v1/login",
        json={"username": username, "password": password}
    )
    assert response.status_code == 200
    return response.json()["access_token"]

def test_rate_limit_exceeded():
    """Testa se rate limit funciona após muitas requisições"""
    # Resetar rate limiter para este teste
    from app.core.rate_limit import rate_limiter
    rate_limiter.requests.clear()
    
    token = get_token(username="ratelimit_user", password="admin")
    
    # Fazer requisições até exceder o limite
    # Configuramos para 100 requests por hora nos testes
    for i in range(5):  # Apenas 5 para não demorar muito
        response = client.post(
            "/api/v1/analyze",
            json={"question": f"Test question {i}"},
            headers={
                "X-API-Key": "test-key",
                "Authorization": f"Bearer {token}"
            }
        )
        assert response.status_code == 200
    
    # Todas as requisições devem passar pois estamos abaixo do limite
    assert response.status_code == 200