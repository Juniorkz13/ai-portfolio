from app.core.settings import settings
from app.core.logging import get_logger
from typing import Optional, List

logger = get_logger(__name__)

def _get_client():
    """Retorna cliente Gemini configurado"""
    try:
        import google.generativeai as genai
        
        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY não configurada")
            return None
        
        genai.configure(api_key=settings.gemini_api_key)
        return genai
    
    except Exception as e:
        logger.error(f"Erro ao configurar Gemini: {str(e)}")
        return None

def list_available_models() -> List[str]:
    """Lista modelos disponíveis na API"""
    try:
        genai = _get_client()
        if not genai:
            return ["Nenhum modelo disponível"]
        
        models = genai.list_models()
        available = []
        for m in models:
            # Filtrar apenas modelos que suportam generateContent
            if "generateContent" in m.supported_generation_methods:
                available.append(m.name)
        
        logger.info(f"Modelos disponíveis: {available}")
        return available
    
    except Exception as e:
        logger.error(f"Erro ao listar modelos: {str(e)}")
        return ["Erro ao listar modelos"]

def get_best_available_model() -> str:
    """Retorna o melhor modelo disponível"""
    try:
        genai = _get_client()
        if not genai:
            return "gemini-flash-latest"
        
        models = genai.list_models()
        
        # Ordem de preferência - modelos mais novos primeiro
        preferred = [
            "gemini-flash-latest",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash", 
            "gemini-pro",
            "gemini-1.0-pro"
        ]
        
        available_names = [m.name for m in models if "generateContent" in m.supported_generation_methods]
        
        logger.info(f"Modelos disponíveis no sistema: {available_names}")
        
        for pref in preferred:
            for available in available_names:
                if pref in available:
                    logger.info(f"Usando modelo: {available}")
                    return available
        
        # Se nenhum preferido encontrado, usar o primeiro disponível
        if available_names:
            logger.info(f"Usando modelo padrão: {available_names[0]}")
            return available_names[0]
        
        return "gemini-flash-latest"
    
    except Exception as e:
        logger.error(f"Erro ao obter melhor modelo: {str(e)}")
        return "gemini-flash-latest"

def generate_text(prompt: str, model: str = None) -> str:
    """Gera texto usando Google Gemini com modelo Flash"""
    try:
        genai = _get_client()
        
        if not genai:
            return "Modelo Gemini não disponível - API key não configurada"
        
        # Usar modelo especificado ou obter o melhor disponível
        model_name = model or get_best_available_model()
        logger.info(f"Gerando com modelo: {model_name}")
        
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        if hasattr(response, 'text'):
            text = response.text
            logger.info(f"Resposta gerada com sucesso ({len(text)} caracteres)")
            return text
        
        return str(response)
    
    except Exception as e:
        logger.error(f"Erro ao gerar com Gemini: {str(e)}")
        return f"Erro ao gerar resposta: {str(e)}"