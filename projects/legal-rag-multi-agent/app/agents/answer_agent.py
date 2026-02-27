from typing import Dict, Any
from app.core.logging import get_logger
from app.llm.gemini_client import generate_text
import time

logger = get_logger(__name__)

class AnswerAgent:
    """Agente que gera a resposta final usando Gemini"""
    
    def __init__(self):
        self.name = "AnswerAgent"
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Gera resposta usando Gemini baseado na análise completa"""
        start_time = time.time()
        
        question = state.get("question", "")
        documents = state.get("documents", [])
        risk_level = state.get("risk_level", "médio")
        domain = state.get("domain", "Geral")
        has_conflicts = state.get("has_conflicts", False)
        is_ambiguous = state.get("is_ambiguous", False)
        missing_info = state.get("missing_info", [])
        
        # Construir contexto para o Gemini
        docs_text = "\n".join([f"- {doc}" for doc in documents[:5]]) if documents else "Nenhum documento fornecido"
        
        # Prompt simples e direto - deixar Gemini responder
        prompt = f"""Você é um assistente jurídico brasileiro especializado em fornecer orientações legais.

**Pergunta do usuário:** {question}

**Domínio jurídico identificado:** {domain}

**Documentos de referência:**
{docs_text}

**Contexto da análise:**
- Nível de risco: {risk_level}
- Há conflitos normativos: {"Sim" if has_conflicts else "Não"}
- Pergunta ambígua: {"Sim" if is_ambiguous else "Não"}
- Informações faltantes: {', '.join(missing_info) if missing_info else "Nenhuma"}

**Instrução:**
Responda a pergunta de forma clara, objetiva e profissional. 
- Cite a legislação brasileira aplicável
- Explique de forma acessível
- Indique próximos passos práticos
- SEMPRE termine com um aviso de que esta é uma orientação preliminar e que deve consultar um advogado para seu caso específico

Responda em português brasileiro."""

        try:
            answer = generate_text(prompt)
            
            # Se vazio ou erro, retornar mensagem
            if not answer or len(answer.strip()) < 20:
                answer = self._get_generic_response(question, domain)
            
        except Exception as e:
            logger.error(f"Erro ao gerar resposta com Gemini: {e}")
            answer = self._get_generic_response(question, domain)
        
        disclaimer = self._generate_disclaimer(risk_level, has_conflicts, is_ambiguous)
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            "answer_generated",
            extra={
                "extra": {
                    "agent": self.name,
                    "domain": domain,
                    "answer_length": len(answer),
                    "risk_level": risk_level,
                    "duration_ms": duration_ms
                }
            }
        )
        
        return {
            "answer": answer,
            "disclaimer": disclaimer
        }
    
    def _get_generic_response(self, question: str, domain: str) -> str:
        """Resposta genérica quando Gemini falha"""
        return f"""Desculpe, não consegui gerar uma resposta detalhada no momento.

**Sua pergunta:** {question}

**Domínio:** {domain}

Por favor, tente novamente ou consulte um advogado especializado em {domain} para uma análise completa do seu caso."""
    
    def _generate_disclaimer(self, risk_level: str, has_conflicts: bool, is_ambiguous: bool) -> str:
        """Gera disclaimer adequado"""
        base = "⚠️ **AVISO IMPORTANTE:** Esta é uma orientação jurídica preliminar gerada por IA."
        
        warnings = []
        
        if risk_level in ["alto", "high"]:
            warnings.append("🔴 **Risco Alto:** Consulte imediatamente um advogado especializado")
        elif risk_level in ["médio", "medium"]:
            warnings.append("🟡 **Risco Médio:** Recomenda-se consulta com advogado")
        
        if has_conflicts:
            warnings.append("⚖️ Há conflitos entre normas identificadas - análise jurídica especializada necessária")
        
        if is_ambiguous:
            warnings.append("❓ A questão é ambígua - detalhes adicionais podem alterar a análise")
        
        warnings.append("👨‍⚖️ **Consulte um advogado registrado na OAB** para análise específica do seu caso")
        
        return base + "\n\n" + "\n".join(warnings)