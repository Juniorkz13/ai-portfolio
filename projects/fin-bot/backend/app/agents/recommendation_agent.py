from __future__ import annotations

from decimal import Decimal

from app.agents.base import BaseAgent
from app.agents.prompts import RECOMMENDATION_AGENT_SYSTEM_PROMPT
from app.agents.types import RecommendationAgentInput, RecommendationAgentOutput


class RecommendationAgent(BaseAgent[RecommendationAgentInput, RecommendationAgentOutput]):
    name = "RecommendationAgent"
    system_prompt = RECOMMENDATION_AGENT_SYSTEM_PROMPT
    allowed_tools: tuple[str, ...] = ("get_month_summary", "get_category_summary")

    def run(self, payload: RecommendationAgentInput) -> RecommendationAgentOutput:
        recommendations: list[str] = []

        if payload.balance < Decimal("0"):
            recommendations.append(
                "Reduza gastos variaveis nas categorias mais altas ate o saldo voltar a ficar positivo."
            )
        else:
            recommendations.append(
                "Reserve parte do saldo positivo para uma meta ou fundo de emergencia."
            )

        if payload.expenses_by_category:
            top_category = max(payload.expenses_by_category, key=lambda item: item.total)
            recommendations.append(
                f"Defina um teto semanal para {top_category.category}, que hoje concentra a maior despesa."
            )

        if payload.insights:
            recommendations.append("Revise semanalmente os padroes identificados para agir antes do fim do mes.")

        summary_text = " ".join(recommendations[:2]) if recommendations else "Nenhuma recomendacao disponivel."

        return RecommendationAgentOutput(
            recommendations=recommendations,
            summary_text=summary_text,
        )
