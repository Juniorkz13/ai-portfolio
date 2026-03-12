from __future__ import annotations

import unicodedata

from app.agents.base import BaseAgent
from app.agents.prompts import CATEGORIZATION_AGENT_SYSTEM_PROMPT
from app.agents.types import CategorizationAgentInput, CategorizationAgentOutput
from app.models.enums import TransactionType


class CategorizationAgent(BaseAgent[CategorizationAgentInput, CategorizationAgentOutput]):
    name = "CategorizationAgent"
    system_prompt = CATEGORIZATION_AGENT_SYSTEM_PROMPT
    allowed_tools: tuple[str, ...] = ("get_category_summary",)

    CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
        "alimentação": (
            "almoco",
            "almoço",
            "jantar",
            "janta",
            "lanche",
            "restaurante",
            "padaria",
            "cafe",
            "café",
        ),
        "transporte": ("uber", "99", "taxi", "metro", "metrô", "onibus", "ônibus"),
        "moradia": ("aluguel", "condominio", "luz", "agua", "internet"),
        "lazer": ("cinema", "show", "bar", "jogo", "streaming"),
        "saúde": ("farmacia", "farmácia", "medico", "médico", "consulta", "plano de saude"),
        "educação": ("curso", "faculdade", "livro", "mensalidade"),
        "outros": (),
    }
    INCOME_ALIASES: dict[str, tuple[str, ...]] = {
        "salário": ("salario", "salário", "holerite", "folha"),
        "freelance": ("freelance", "freela", "projeto", "job"),
        "reembolso": ("reembolso",),
        "investimentos": ("dividendo", "dividendos", "rendimento", "rendimentos", "juros"),
        "pix": ("pix",),
        "outros": (),
    }

    def run(self, payload: CategorizationAgentInput) -> CategorizationAgentOutput:
        if payload.transaction_type == TransactionType.INCOME:
            candidate = self._normalize_text(
                (payload.suggested_category or payload.description or "").casefold()
            )
            for category, aliases in self.INCOME_ALIASES.items():
                if candidate == self._normalize_text(category):
                    return CategorizationAgentOutput(category=category)
                if any(self._normalize_text(alias) in candidate for alias in aliases):
                    return CategorizationAgentOutput(category=category)
            return CategorizationAgentOutput(category=payload.suggested_category or "outros")

        candidate = self._normalize_text((payload.suggested_category or payload.description or "").casefold())

        for category, aliases in self.CATEGORY_ALIASES.items():
            if candidate == self._normalize_text(category):
                return CategorizationAgentOutput(category=category)
            if any(self._normalize_text(alias) in candidate for alias in aliases):
                return CategorizationAgentOutput(category=category)

        return CategorizationAgentOutput(category="outros")

    def _normalize_text(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        return "".join(char for char in normalized if not unicodedata.combining(char))
