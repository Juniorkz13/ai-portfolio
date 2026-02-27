from typing import Dict, Any, List
from app.core.logging import get_logger
import time

logger = get_logger(__name__)

class LegalInterpreterAgent:
    """Agente que interpreta questões jurídicas e identifica o domínio"""
    
    # Palavras-chave para identificar domínios jurídicos
    DOMAIN_KEYWORDS = {
        "Trabalhista": ["trabalho", "emprego", "demissão", "férias", "salário", "jornada", "CLT", "trabalhista", "contrato de trabalho"],
        "Civil": ["contrato", "indenização", "dano", "obrigação", "responsabilidade civil", "patrimônio"],
        "Consumidor": ["consumidor", "defeito", "produto", "serviço", "CDC", "garantia", "compra", "venda"],
        "Família": ["divórcio", "pensão alimentícia", "guarda", "casamento", "filhos", "herança", "testamento"],
        "Penal": ["crime", "processo criminal", "acusação", "pena", "código penal", "delegacia"],
        "Tributário": ["imposto", "tributo", "ICMS", "ISS", "IR", "fiscal", "taxa"],
        "Empresarial": ["empresa", "sociedade", "contrato social", "MEI", "CNPJ", "sócio"],
        "Previdenciário": ["aposentadoria", "INSS", "benefício", "auxílio", "contribuição"],
        "Imobiliário": ["imóvel", "aluguel", "locação", "propriedade", "usucapião", "condomínio"]
    }
    
    def __init__(self):
        self.name = "LegalInterpreterAgent"
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Interpreta a questão e identifica domínio e ambiguidades"""
        start_time = time.time()
        
        question = state.get("question", "").lower()
        documents = state.get("documents", [])
        
        # Identificar domínio
        domain = self._identify_domain(question)
        
        # Detectar ambiguidades
        is_ambiguous = self._detect_ambiguity(question, documents)
        
        # Identificar informações faltantes
        missing_info = self._identify_missing_info(question)
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            "legal_interpretation_completed",
            extra={
                "extra": {
                    "agent": self.name,
                    "domain": domain,
                    "is_ambiguous": is_ambiguous,
                    "missing_info_count": len(missing_info),
                    "duration_ms": duration_ms
                }
            }
        )
        
        return {
            "domain": domain,
            "is_ambiguous": is_ambiguous,
            "missing_info": missing_info
        }
    
    def _identify_domain(self, question: str) -> str:
        """Identifica o domínio jurídico da questão"""
        domain_scores = {}
        
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in question)
            if score > 0:
                domain_scores[domain] = score
        
        if not domain_scores:
            return "Geral"
        
        return max(domain_scores, key=domain_scores.get)
    
    def _detect_ambiguity(self, question: str, documents: List[str]) -> bool:
        """Detecta ambiguidades na questão"""
        ambiguity_indicators = [
            "talvez", "pode ser", "acho que", "não tenho certeza",
            "será que", "depende", "não sei", "??"
        ]
        
        has_ambiguous_terms = any(term in question for term in ambiguity_indicators)
        has_no_documents = len(documents) == 0
        
        return has_ambiguous_terms or has_no_documents
    
    def _identify_missing_info(self, question: str) -> List[str]:
        """Identifica informações que podem estar faltando"""
        missing = []
        
        # Se a pergunta é muito curta, pode estar faltando contexto
        if len(question) < 20:
            missing.append("Contexto adicional")
        
        # Verificar se há datas mencionadas
        if not any(char.isdigit() for char in question):
            missing.append("Datas ou períodos específicos")
        
        # Verificar se há valores ou quantidades
        if "quanto" not in question and "valor" not in question and "quanto" not in question:
            if "R$" not in question and "$" not in question:
                missing.append("Valores ou quantidades específicas")
        
        return missing[:2]  # Máximo 2 itens