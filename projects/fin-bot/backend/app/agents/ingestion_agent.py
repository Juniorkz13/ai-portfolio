from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation

from app.agents.base import BaseAgent
from app.agents.prompts import INGESTION_AGENT_SYSTEM_PROMPT
from app.agents.types import IngestionAgentInput, IngestionAgentOutput
from app.models.enums import TransactionType

AMOUNT_PATTERN = re.compile(r"(?P<amount>\d+(?:[.,]\d{1,2})?)")


class IngestionAgent(BaseAgent[IngestionAgentInput, IngestionAgentOutput]):
    name = "IngestionAgent"
    system_prompt = INGESTION_AGENT_SYSTEM_PROMPT
    allowed_tools: tuple[str, ...] = ()

    CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
        "transporte": ("uber", "99", "taxi", "onibus", "ônibus", "metro", "metrô"),
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
        "moradia": ("aluguel", "condominio", "energia", "agua", "internet"),
        "lazer": ("cinema", "show", "bar", "viagem", "streaming"),
        "saúde": ("farmacia", "farmácia", "medico", "médico", "consulta", "exame"),
        "educação": ("curso", "faculdade", "livro", "mensalidade"),
    }

    def run(self, payload: IngestionAgentInput) -> IngestionAgentOutput:
        message = payload.message.strip()
        message_lower = message.casefold()
        normalized_message = self._normalize_text(message_lower)

        transaction_type = self._extract_transaction_type(normalized_message)
        amount = self._extract_amount(message_lower)
        extracted_date = self._extract_date(normalized_message, payload.reference_date)
        category = self._extract_category(message_lower, normalized_message)
        description = self._extract_description(message)

        return IngestionAgentOutput(
            type=transaction_type,
            amount=amount,
            category=category,
            description=description,
            date=extracted_date,
        )

    def _extract_transaction_type(self, message: str) -> TransactionType | None:
        if any(keyword in message for keyword in ("recebi", "ganhei", "salario", "pix recebido")):
            return TransactionType.INCOME
        if any(keyword in message for keyword in ("gastei", "paguei", "comprei", "debito")):
            return TransactionType.EXPENSE
        return None

    def _extract_amount(self, message: str) -> Decimal | None:
        match = AMOUNT_PATTERN.search(message)
        if not match:
            return None

        raw_amount = match.group("amount").replace(".", "").replace(",", ".")
        try:
            return Decimal(raw_amount)
        except InvalidOperation:
            return None

    def _extract_date(self, message: str, reference_date: date | None) -> date | None:
        base_date = reference_date or date.today()

        if "hoje" in message:
            return base_date
        if "ontem" in message:
            return base_date - timedelta(days=1)
        if "anteontem" in message:
            return base_date - timedelta(days=2)

        iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", message)
        if iso_match:
            return date.fromisoformat(iso_match.group(1))

        br_match = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", message)
        if br_match:
            day, month, year = br_match.groups()
            return date(int(year), int(month), int(day))

        return None

    def _extract_category(self, message: str, normalized_message: str) -> str | None:
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(
                self._normalize_text(keyword) in normalized_message
                or keyword in message
                for keyword in keywords
            ):
                return category
        return None

    def _extract_description(self, message: str) -> str | None:
        cleaned = re.sub(r"\b(gastei|paguei|comprei|recebi|ganhei)\b", "", message, flags=re.I)
        cleaned = re.sub(r"\b(\d+(?:[.,]\d{1,2})?)\b", "", cleaned)
        cleaned = re.sub(r"\b(hoje|ontem|anteontem)\b", "", cleaned, flags=re.I)
        cleaned = re.sub(r"^\s*(no|na|de|do|da|dos|das)\s+", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s+(no|na|de|do|da|dos|das)\s+", " ", cleaned, flags=re.I)
        description = " ".join(cleaned.split()).strip(" -")
        return description or None

    def _normalize_text(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        return "".join(char for char in normalized if not unicodedata.combining(char))
