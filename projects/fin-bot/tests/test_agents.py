from __future__ import annotations

from datetime import date
from decimal import Decimal

from app.agents import (
    AnalyticsAgent,
    CategorizationAgent,
    IngestionAgent,
    RecommendationAgent,
    RouterAgent,
)
from app.agents.types import (
    AnalyticsAgentInput,
    CategorizationAgentInput,
    IngestionAgentInput,
    IntentType,
    RecommendationAgentInput,
    RouterAgentInput,
    TransactionAnalyticsItem,
)
from app.models.enums import TransactionType


def test_router_agent_detects_expense_intent() -> None:
    agent = RouterAgent()
    result = agent.run(RouterAgentInput(message="Gastei 32 no Uber ontem"))
    assert result.intent == IntentType.REGISTER_EXPENSE


def test_router_agent_detects_insights_intent() -> None:
    agent = RouterAgent()
    result = agent.run(RouterAgentInput(message="Qual foi meu maior gasto do mês?"))
    assert result.intent == IntentType.GET_INSIGHTS


def test_ingestion_agent_extracts_transaction_fields() -> None:
    agent = IngestionAgent()
    result = agent.run(
        IngestionAgentInput(
            message="Gastei 32 no Uber ontem",
            reference_date=date(2026, 3, 11),
        )
    )
    assert result.type == TransactionType.EXPENSE
    assert result.amount == Decimal("32")
    assert result.category == "transporte"
    assert result.date == date(2026, 3, 10)


def test_categorization_agent_falls_back_to_known_category() -> None:
    agent = CategorizationAgent()
    result = agent.run(
        CategorizationAgentInput(
            type=TransactionType.EXPENSE,
            description="Corrida de Uber para o trabalho",
        )
    )
    assert result.category == "transporte"


def test_analytics_agent_generates_insights() -> None:
    agent = AnalyticsAgent()
    result = agent.run(
        AnalyticsAgentInput(
            month=3,
            year=2026,
            total_expenses=Decimal("450.00"),
            total_income=Decimal("1000.00"),
            balance=Decimal("550.00"),
            expenses_by_category=[
                TransactionAnalyticsItem(category="transporte", total=Decimal("200.00"))
            ],
        )
    )
    assert result.insights
    assert "03/2026" in result.summary_text


def test_recommendation_agent_generates_actions() -> None:
    agent = RecommendationAgent()
    result = agent.run(
        RecommendationAgentInput(
            total_expenses=Decimal("800.00"),
            total_income=Decimal("700.00"),
            balance=Decimal("-100.00"),
            insights=["As despesas superaram as receitas no periodo."],
            expenses_by_category=[
                TransactionAnalyticsItem(category="alimentacao", total=Decimal("300.00"))
            ],
        )
    )
    assert len(result.recommendations) >= 2
