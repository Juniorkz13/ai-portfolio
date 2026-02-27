from unittest.mock import patch, MagicMock
from app.llm.gemini_client import generate_text

@patch('app.llm.gemini_client._get_client')
def test_generate_text_returns_string(mock_get_client):
    """Testa se generate_text retorna string"""
    # Configurar o mock para retornar a resposta esperada
    mock_response = MagicMock()
    mock_response.text = "Resposta simulada do Gemini"
    
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    
    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model
    
    mock_get_client.return_value = mock_genai
    
    # Chamar a função
    result = generate_text("test prompt")
    
    # Verificar resultado
    assert result == "Resposta simulada do Gemini"
    assert isinstance(result, str)