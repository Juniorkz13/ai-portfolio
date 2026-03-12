from __future__ import annotations

import uuid

from app.agents.analytics_agent import AnalyticsAgent
from app.agents.types import AnalyticsAgentInput, TransactionAnalyticsItem
from app.schemas.summary import MonthSummaryResponse


def build_month_insights(
    *,
    summary: MonthSummaryResponse,
    user_id: uuid.UUID | None,
    month: int,
    year: int,
) -> list[str]:
    agent = AnalyticsAgent()
    result = agent.run(
        AnalyticsAgentInput(
            user_id=user_id,
            month=month,
            year=year,
            total_expenses=summary.total_expenses,
            total_income=summary.total_income,
            balance=summary.balance,
            expenses_by_category=[
                TransactionAnalyticsItem(category=item.category, total=item.total)
                for item in summary.expenses_by_category
            ],
        )
    )
    return result.insights
