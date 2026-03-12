from __future__ import annotations

from decimal import Decimal

from app.agents.base import BaseAgent
from app.agents.prompts import ANALYTICS_AGENT_SYSTEM_PROMPT
from app.agents.types import AnalyticsAgentInput, AnalyticsAgentOutput


class AnalyticsAgent(BaseAgent[AnalyticsAgentInput, AnalyticsAgentOutput]):
    name = "AnalyticsAgent"
    system_prompt = ANALYTICS_AGENT_SYSTEM_PROMPT
    allowed_tools: tuple[str, ...] = ("get_month_summary", "get_category_summary")

    def run(self, payload: AnalyticsAgentInput) -> AnalyticsAgentOutput:
        insights: list[str] = []

        if payload.total_income == Decimal("0"):
            insights.append("Nao houve receitas registradas no periodo analisado.")
        elif payload.balance < Decimal("0"):
            insights.append("As despesas superaram as receitas no periodo.")
        else:
            insights.append("O saldo do periodo permaneceu positivo.")

        if payload.expenses_by_category:
            top_category = max(payload.expenses_by_category, key=lambda item: item.total)
            insights.append(
                f"A categoria com maior gasto foi {top_category.category}, totalizando {top_category.total}."
            )

        if payload.total_expenses == Decimal("0"):
            insights.append("Nao ha despesas registradas para o periodo informado.")

        month_label = (
            f"{payload.month:02d}/{payload.year}"
            if payload.month is not None and payload.year is not None
            else "periodo selecionado"
        )
        summary_text = (
            f"No {month_label}, as receitas somaram {payload.total_income}, "
            f"as despesas ficaram em {payload.total_expenses} "
            f"e o saldo foi {payload.balance}."
        )

        return AnalyticsAgentOutput(insights=insights, summary_text=summary_text)
