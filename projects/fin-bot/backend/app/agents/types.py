from __future__ import annotations

import uuid
from datetime import date as DateValue
from decimal import Decimal as DecimalValue
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import TransactionType


class AgentModel(BaseModel):
    model_config = ConfigDict(use_enum_values=False, arbitrary_types_allowed=True)


class IntentType(str, Enum):
    REGISTER_EXPENSE = "registrar_despesa"
    REGISTER_INCOME = "registrar_receita"
    GET_SUMMARY = "consultar_resumo"
    GET_INSIGHTS = "consultar_insights"
    GET_HISTORY = "consultar_historico"
    IMPORT_CSV = "importar_csv"
    OTHER = "outro"


class RouterAgentInput(AgentModel):
    message: str


class RouterAgentOutput(AgentModel):
    intent: IntentType


class IngestionAgentInput(AgentModel):
    message: str
    reference_date: DateValue | None = None


class IngestionAgentOutput(AgentModel):
    type: TransactionType | None = None
    amount: DecimalValue | None = None
    category: str | None = None
    description: str | None = None
    date: DateValue | None = None


class CategorizationAgentInput(AgentModel):
    description: str | None = None
    suggested_category: str | None = None
    transaction_type: TransactionType = Field(alias="type")


class CategorizationAgentOutput(AgentModel):
    category: str


class TransactionAnalyticsItem(AgentModel):
    category: str
    total: DecimalValue


class AnalyticsAgentInput(AgentModel):
    user_id: uuid.UUID | None = None
    month: int | None = None
    year: int | None = None
    total_expenses: DecimalValue
    total_income: DecimalValue
    balance: DecimalValue
    expenses_by_category: list[TransactionAnalyticsItem] = Field(default_factory=list)


class AnalyticsAgentOutput(AgentModel):
    insights: list[str]
    summary_text: str


class RecommendationAgentInput(AgentModel):
    user_id: uuid.UUID | None = None
    month: int | None = None
    year: int | None = None
    insights: list[str] = Field(default_factory=list)
    total_expenses: DecimalValue
    total_income: DecimalValue
    balance: DecimalValue
    expenses_by_category: list[TransactionAnalyticsItem] = Field(default_factory=list)


class RecommendationAgentOutput(AgentModel):
    recommendations: list[str]
    summary_text: str


class AgentResult(AgentModel):
    agent_name: str
    payload: dict[str, Any]
