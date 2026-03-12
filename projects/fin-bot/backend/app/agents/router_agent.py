from __future__ import annotations

import unicodedata

from app.agents.base import BaseAgent
from app.agents.prompts import ROUTER_AGENT_SYSTEM_PROMPT
from app.agents.types import IntentType, RouterAgentInput, RouterAgentOutput


class RouterAgent(BaseAgent[RouterAgentInput, RouterAgentOutput]):
    name = "RouterAgent"
    system_prompt = ROUTER_AGENT_SYSTEM_PROMPT
    allowed_tools: tuple[str, ...] = ()

    def run(self, payload: RouterAgentInput) -> RouterAgentOutput:
        message = self._normalize_text(payload.message.casefold())

        if "csv" in message or "planilha" in message or "arquivo" in message:
            intent = IntentType.IMPORT_CSV
        elif any(
            keyword in message
            for keyword in (
                "insight financeiro",
                "insights financeiros",
                "me de um insight",
                "me de um insight financeiro",
                "analise minhas financas",
                "analise minhas financas deste mes",
                "como estao meus gastos este mes",
                "como estao meus gastos esse mes",
                "qual foi meu maior gasto do mes",
                "qual foi a maior categoria de gasto",
            )
        ):
            intent = IntentType.GET_INSIGHTS
        elif any(
            keyword in message
            for keyword in (
                "resumo",
                "saldo",
                "fechamento",
                "quanto gastei",
                "quanto eu gastei",
                "esse mes",
                "este mes",
            )
        ):
            intent = IntentType.GET_SUMMARY
        elif any(
            keyword in message
            for keyword in (
                "historico",
                "ultimas",
                "ultimos",
                "ultimas transacoes",
                "ultimos gastos",
                "ultimas despesas",
                "transacoes",
                "movimentacoes",
            )
        ):
            intent = IntentType.GET_HISTORY
        elif any(keyword in message for keyword in ("recebi", "ganhei", "entrada", "receita")):
            intent = IntentType.REGISTER_INCOME
        elif any(
            keyword in message
            for keyword in ("gastei", "paguei", "comprei", "despesa", "gasto")
        ):
            intent = IntentType.REGISTER_EXPENSE
        else:
            intent = IntentType.OTHER

        return RouterAgentOutput(intent=intent)

    def _normalize_text(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        return "".join(char for char in normalized if not unicodedata.combining(char))
