import time
from typing import Dict
from app.core.settings import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class RateLimiter:
    """Rate limiter simples baseado em tempo"""
    
    def __init__(self):
        self.requests: Dict[str, list] = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Verifica se cliente pode fazer requisição"""
        now = time.time()
        window_start = now - settings.rate_limit_window_seconds
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove requests fora da janela
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        
        # Verifica limite
        if len(self.requests[client_id]) >= settings.rate_limit_requests:
            logger.warning(f"Rate limit exceeded for {client_id}")
            return False
        
        # Adiciona nova requisição
        self.requests[client_id].append(now)
        return True

# Instância global
rate_limiter = RateLimiter()