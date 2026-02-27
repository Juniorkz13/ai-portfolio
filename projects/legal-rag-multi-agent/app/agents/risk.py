import time
from app.core.logging import get_logger
from app.agents.base import BaseAgent

logger = get_logger(__name__)


class RiskAgent(BaseAgent):
    """Agente de avaliação de risco jurídico."""

    def run(self, input_data: dict) -> dict:
        start = time.time()

        try:
            has_conflict = input_data.get("has_conflict", False)
            conflicts = input_data.get("conflicts", [])
            missing_info = input_data.get("missing_information", [])

            risk_factors = []

            if has_conflict:
                risk_factors.extend(conflicts)

            if missing_info:
                risk_factors.append(
                    "Informações essenciais ausentes: " + ", ".join(missing_info)
                )

            if has_conflict and missing_info:
                risk_level = "alto"
                recommendation = (
                    "Cenário jurídico complexo. Recomenda-se análise especializada."
                )
            elif has_conflict or missing_info:
                risk_level = "medio"
                recommendation = (
                    "Existem fatores de risco. Avaliação jurídica detalhada é recomendada."
                )
            else:
                risk_level = "baixo"
                recommendation = (
                    "Cenário jurídico aparentemente simples, sem conflitos detectados."
                )

            result = {
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "recommendation": recommendation,
            }

            logger.info(
                "risk_evaluated",
                extra={
                    "extra": {
                        "request_id": input_data.get("request_id"),
                        "agent": "RiskAgent",
                        "risk_level": risk_level,
                        "duration_ms": int((time.time() - start) * 1000),
                    }
                },
            )

            return result

        except Exception as e:
            logger.error(
                "risk_evaluation_failed",
                extra={
                    "extra": {
                        "request_id": input_data.get("request_id"),
                        "agent": "RiskAgent",
                        "error": str(e),
                    }
                },
            )
            raise

    def analyze(self, context: str, question: str, request_id: str = None) -> dict:
        start_time = time.time()

        try:
            # Análise de risco baseada em múltiplos fatores
            risk_level = "baixo"
            risk_score = 0.0
            key_factors = []
            recommendations = []

            # 1. Análise de termos de risco
            risk_terms = {
                "alto": ["crime", "penal", "prisão", "reclusão", "condenação", "execução"],
                "medio": [
                    "conflito",
                    "litígio",
                    "processo",
                    "ação judicial",
                    "divergência",
                    "recurso",
                    "apelação",
                    "contestação",
                    "impugnação",
                ],
                "baixo": ["dúvida", "orientação", "consulta", "esclarecimento"],
            }

            text_to_analyze = f"{context} {question}".lower()

            # Detecta termos de alto risco
            high_risk_found = [
                term for term in risk_terms["alto"] if term in text_to_analyze
            ]
            medium_risk_found = [
                term for term in risk_terms["medio"] if term in text_to_analyze
            ]

            # 2. Análise de complexidade (tamanho e estrutura do texto)
            complexity_score = 0.0
            word_count = len(context.split())

            if word_count > 1000:
                complexity_score = 0.3
                key_factors.append("Contexto extenso e complexo")
            elif word_count > 500:
                complexity_score = 0.2
                key_factors.append("Contexto moderadamente complexo")

            # 3. Análise de múltiplas questões ou aspectos
            question_indicators = [
                "e também",
                "além disso",
                "outra dúvida",
                "ainda",
                "também",
            ]
            if any(indicator in question.lower() for indicator in question_indicators):
                complexity_score += 0.1
                key_factors.append("Questão com múltiplos aspectos")

            # 4. Cálculo do score final
            if high_risk_found:
                risk_score = 0.8 + complexity_score
                risk_level = "alto"
                key_factors.extend(
                    [f"Termo crítico: '{term}'" for term in high_risk_found[:3]]
                )
                recommendations.extend(
                    [
                        "Consulta urgente com advogado especializado recomendada",
                        "Considere buscar assessoria jurídica imediata",
                        "Análise detalhada de possíveis consequências legais necessária",
                    ]
                )
            elif medium_risk_found:
                risk_score = 0.5 + complexity_score
                risk_level = "medio"
                key_factors.extend(
                    [f"Termo de risco: '{term}'" for term in medium_risk_found[:3]]
                )
                recommendations.extend(
                    [
                        "Recomenda-se análise jurídica detalhada",
                        "Avaliação especializada pode ser necessária",
                        "Considere documentar todas as evidências relevantes",
                    ]
                )
            else:
                risk_score = 0.2 + complexity_score
                risk_level = "baixo"
                recommendations.append("Análise preliminar concluída - cenário aparentemente simples")

            # 5. Ajuste baseado em incerteza
            uncertainty_terms = ["não sei", "talvez", "pode ser", "acho que", "incerto"]
            if any(term in text_to_analyze for term in uncertainty_terms):
                risk_score = min(risk_score + 0.1, 1.0)
                key_factors.append("Informações incertas ou incompletas detectadas")
                recommendations.append("Buscar documentação adicional para esclarecer incertezas")

            # Normaliza o score entre 0 e 1
            risk_score = min(risk_score, 1.0)

            # Adiciona recomendação padrão se não houver nenhuma
            if not recommendations:
                recommendations.append("Nenhuma recomendação específica neste momento")

            result = {
                "risk_level": risk_level,
                "risk_score": risk_score,
                "key_factors": key_factors,
                "recommendations": recommendations,
            }

            duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "risk_analysis_success",
                extra={
                    "extra": {
                        "request_id": request_id,
                        "agent": "RiskAgent",
                        "risk_level": risk_level,
                        "risk_score": risk_score,
                        "num_key_factors": len(key_factors),
                        "num_recommendations": len(recommendations),
                        "duration_ms": duration_ms,
                    }
                },
            )

            return result

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            logger.error(
                "risk_analysis_failed",
                extra={
                    "extra": {
                        "request_id": request_id,
                        "agent": "RiskAgent",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "duration_ms": duration_ms,
                    }
                },
                exc_info=True,
            )

            raise