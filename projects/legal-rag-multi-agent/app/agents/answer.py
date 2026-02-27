from app.core.agent import BaseAgent
import time
from typing import Dict, Any, List
from app.core.logging import get_logger

logger = get_logger(__name__)


class AnswerAgent(BaseAgent):
    """Agente de geração de respostas jurídicas baseadas em RAG."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def run(self, input_data: dict) -> dict:
        start_time = time.time()

        try:
            question = input_data.get("question", "")
            documents = input_data.get("documents", [])
            request_id = input_data.get("request_id")
            
            # Gera resposta baseada em documentos
            answer = self._generate_answer(question, documents)
            
            # Extrai referências citadas
            cited_references = self._extract_cited_references(answer)
            
            # Calcula confiança da resposta
            confidence_score = self._calculate_confidence(answer, documents)
            
            result = {
                "answer": answer,
                "cited_references": cited_references,
                "confidence_score": confidence_score,
                "num_sources": len(documents),
                "disclaimer": self._generate_disclaimer(confidence_score, len(documents)),
            }

            duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "answer_generated",
                extra={
                    "extra": {
                        "request_id": request_id,
                        "agent": "AnswerAgent",
                        "answer_length": len(answer),
                        "num_cited_references": len(cited_references),
                        "confidence_score": confidence_score,
                        "num_sources": len(documents),
                        "duration_ms": duration_ms,
                    }
                },
            )

            return result

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            logger.error(
                "answer_generation_failed",
                extra={
                    "extra": {
                        "request_id": input_data.get("request_id"),
                        "agent": "AnswerAgent",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "duration_ms": duration_ms,
                    }
                },
                exc_info=True,
            )

            raise

    def _generate_answer(self, question: str, documents: List[Dict[str, Any]]) -> str:
        """
        Gera uma resposta jurídica detalhada baseada na questão e documentos.
        
        Args:
            question: Pergunta jurídica do usuário
            documents: Lista de documentos relevantes recuperados
            
        Returns:
            Resposta estruturada e bem fundamentada
        """
        if not documents:
            return "Desculpe, não foi possível encontrar documentos relevantes para responder sua questão."
        
        # 1. Estrutura a resposta em seções
        answer_parts = []
        
        # Seção 1: Resumo executivo
        summary = self._generate_summary(question, documents)
        answer_parts.append(f"**Resumo:**\n{summary}\n")
        
        # Seção 2: Análise detalhada
        analysis = self._generate_detailed_analysis(question, documents)
        answer_parts.append(f"**Análise Detalhada:**\n{analysis}\n")
        
        # Seção 3: Recomendações
        recommendations = self._generate_recommendations(question, documents)
        answer_parts.append(f"**Recomendações:**\n{recommendations}\n")
        
        # Seção 4: Referências utilizadas
        references_section = self._generate_references_section(documents)
        answer_parts.append(f"**Referências:**\n{references_section}")
        
        return "\n".join(answer_parts)

    def _generate_summary(self, question: str, documents: List[Dict[str, Any]]) -> str:
        """Gera um resumo executivo da resposta."""
        # Extrai os pontos principais dos documentos
        key_points = []
        
        for doc in documents[:2]:  # Usa os 2 primeiros documentos
            content = doc.get("text") or doc.get("content", "")
            
            # Extrai primeira frase ou parágrafo
            sentences = content.split(". ")
            if sentences:
                key_points.append(sentences[0])
        
        if key_points:
            return " ".join(key_points[:2])
        
        return "Baseado nos documentos jurídicos analisados, foi possível identificar informações relevantes para sua questão."

    def _generate_detailed_analysis(self, question: str, documents: List[Dict[str, Any]]) -> str:
        """Gera uma análise detalhada baseada nos documentos."""
        analysis_parts = []
        
        for idx, doc in enumerate(documents, 1):
            content = doc.get("text") or doc.get("content", "")
            source = doc.get("source", "Documento sem identificação")
            
            # Limita o tamanho do conteúdo
            content_preview = content[:300] + "..." if len(content) > 300 else content
            
            analysis_parts.append(
                f"{idx}. **{source}**\n"
                f"   {content_preview}\n"
            )
        
        return "\n".join(analysis_parts)

    def _generate_recommendations(self, question: str, documents: List[Dict[str, Any]]) -> str:
        """Gera recomendações baseadas na análise."""
        recommendations = []
        
        # Analisa o tipo de questão
        question_lower = question.lower()
        
        # Recomendações baseadas no conteúdo dos documentos
        if any(term in question_lower for term in ["crime", "penal", "prisão"]):
            recommendations.append("• Procure um advogado especialista em direito penal para análise específica do seu caso")
            recommendations.append("• Reúna toda documentação relevante para apresentação ao profissional")
            recommendations.append("• Esteja preparado para discussões sobre possibilidades legais e consequências")
        
        elif any(term in question_lower for term in ["trabalho", "emprego", "demissão"]):
            recommendations.append("• Consulte um advogado especializado em direito trabalhista")
            recommendations.append("• Revise todos os documentos relacionados ao seu contrato de trabalho")
            recommendations.append("• Documente todos os incidentes e comunicações relevantes")
            recommendations.append("• Verifique os prazos para ações trabalhistas cabíveis")
        
        elif any(term in question_lower for term in ["contrato", "civil", "indenização"]):
            recommendations.append("• Procure orientação jurídica especializada em direito civil")
            recommendations.append("• Reúna toda documentação do contrato e comunicações")
            recommendations.append("• Preserve evidências de danos ou descumprimentos")
            recommendations.append("• Considere mediação antes de ações judiciais")
        
        elif any(term in question_lower for term in ["consumidor", "produto", "serviço"]):
            recommendations.append("• Verifique seus direitos como consumidor sob o CDC")
            recommendations.append("• Entre em contato com o fornecedor para resolução")
            recommendations.append("• Procure órgãos de proteção ao consumidor se necessário")
            recommendations.append("• Mantenha registros de compras e comunicações")
        
        else:
            recommendations.append("• Procure orientação jurídica especializada para seu caso específico")
            recommendations.append("• Reúna toda documentação relevante")
            recommendations.append("• Estabeleça cronograma de ações necessárias")
        
        return "\n".join(recommendations)

    def _generate_references_section(self, documents: List[Dict[str, Any]]) -> str:
        """Gera seção de referências e fontes utilizadas."""
        references = []
        
        for idx, doc in enumerate(documents, 1):
            source = doc.get("source", "Fonte desconhecida")
            
            # Extrai metadata se disponível
            metadata = doc.get("metadata", {})
            page = metadata.get("page")
            article = metadata.get("article")
            
            # Formata a referência
            ref_text = f"{idx}. {source}"
            
            if article:
                ref_text += f" - {article}"
            
            if page:
                ref_text += f", página {page}"
            
            references.append(ref_text)
        
        return "\n".join(references)

    def _extract_cited_references(self, answer: str) -> List[str]:
        """Extrai referências citadas na resposta."""
        cited = []
        
        # Identifica padrões de referência
        import re
        
        # Busca por citações de artigos (ex: "Art. 186")
        article_pattern = r'Art\.?\s+\d+'
        articles = re.findall(article_pattern, answer)
        cited.extend(articles)
        
        # Busca por siglas de leis (ex: "CLT", "CC", "CPC")
        law_pattern = r'\b(CLT|CC|CPC|CF|CDC|CP)\b'
        laws = re.findall(law_pattern, answer)
        cited.extend(laws)
        
        # Busca por jurisprudência (ex: "STF", "STJ", "Súmula")
        jurisprudence_pattern = r'\b(STF|STJ|OAB|Súmula)\b'
        jurisprudence = re.findall(jurisprudence_pattern, answer)
        cited.extend(jurisprudence)
        
        # Remove duplicatas mantendo ordem
        cited = list(dict.fromkeys(cited))
        
        return cited

    def _calculate_confidence(self, answer: str, documents: List[Dict[str, Any]]) -> float:
        """
        Calcula um score de confiança da resposta.
        
        Baseado em:
        - Quantidade de documentos disponíveis
        - Relevância dos documentos
        - Quantidade de referências legais citadas
        
        Returns:
            Score entre 0 e 1 (0 = baixa confiança, 1 = alta confiança)
        """
        confidence = 0.5  # Baseline
        
        # 1. Quantidade de documentos
        if len(documents) >= 3:
            confidence += 0.2
        elif len(documents) >= 1:
            confidence += 0.1
        
        # 2. Comprimento da resposta (indica detalhamento)
        if len(answer) > 1000:
            confidence += 0.15
        elif len(answer) > 500:
            confidence += 0.1
        
        # 3. Presença de referências legais
        cited_refs = self._extract_cited_references(answer)
        if len(cited_refs) >= 5:
            confidence += 0.15
        elif len(cited_refs) >= 2:
            confidence += 0.1
        
        # 4. Presença de recomendações estruturadas
        if "**Recomendações:**" in answer:
            confidence += 0.1
        
        # Normaliza para máximo 1.0
        confidence = min(confidence, 1.0)
        
        return round(confidence, 2)

    def _generate_disclaimer(self, confidence_score: float, num_sources: int) -> str:
        """Gera disclaimer baseado na confiança e fontes."""
        disclaimer = "⚠️ **AVISO LEGAL**: "
                
        if confidence_score < 0.6:
            disclaimer += "Esta resposta é baseada em informações limitadas e NÃO substitui orientação jurídica profissional. "
        else:
            disclaimer += "Embora fundamentada em documentos jurídicos, esta resposta NÃO substitui orientação de um advogado especializado. "
                
        disclaimer += "Consulte sempre um profissional qualificado antes de tomar decisões legais."
                
        return disclaimer