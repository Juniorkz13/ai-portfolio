from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Classe base para todos os agentes."""
    
    @abstractmethod
    def run(self, input_data: dict) -> dict:
        """Executa o agente com os dados de entrada."""
        pass