from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Transaction, TransactionType
from app.tools._compat import tool
from app.tools.common import get_month_date_range
from app.tools.types import (
    CategorySummaryItem,
    GetCategorySummaryInput,
    SessionFactory,
)


def get_category_summary(
    session: Session, payload: GetCategorySummaryInput
) -> list[CategorySummaryItem]:
    start_date, end_date = get_month_date_range(payload.year, payload.month)

    stmt = (
        select(Transaction.category, func.coalesce(func.sum(Transaction.amount), 0))
        .where(
            Transaction.user_id == payload.user_id,
            Transaction.type == TransactionType.EXPENSE,
            Transaction.date >= start_date,
            Transaction.date <= end_date,
        )
        .group_by(Transaction.category)
        .order_by(func.sum(Transaction.amount).desc(), Transaction.category.asc())
    )

    return [
        CategorySummaryItem(category=category, total=total)
        for category, total in session.execute(stmt).all()
    ]


def build_get_category_summary_tool(session_factory: SessionFactory):
    @tool("get_category_summary", args_schema=GetCategorySummaryInput)
    def get_category_summary_tool(
        user_id: str, month: int, year: int
    ) -> list[dict[str, object]]:
        payload = GetCategorySummaryInput(user_id=user_id, month=month, year=year)
        with session_factory() as session:
            results = get_category_summary(session, payload)
        return [result.model_dump(mode="json") for result in results]

    return get_category_summary_tool
