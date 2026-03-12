from __future__ import annotations

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.models import Transaction, TransactionType
from app.tools._compat import tool
from app.tools.common import decimal_zero, get_month_date_range
from app.tools.types import (
    CategorySummaryItem,
    MonthSummaryInput,
    MonthSummaryOutput,
    SessionFactory,
)


def get_month_summary(session: Session, payload: MonthSummaryInput) -> MonthSummaryOutput:
    start_date, end_date = get_month_date_range(payload.year, payload.month)

    totals_stmt = select(
        func.coalesce(
            func.sum(
                case((Transaction.type == TransactionType.EXPENSE, Transaction.amount), else_=0)
            ),
            0,
        ),
        func.coalesce(
            func.sum(
                case((Transaction.type == TransactionType.INCOME, Transaction.amount), else_=0)
            ),
            0,
        ),
    ).where(
        Transaction.user_id == payload.user_id,
        Transaction.date >= start_date,
        Transaction.date <= end_date,
    )

    total_expenses, total_income = session.execute(totals_stmt).one()

    categories_stmt = (
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
    expenses_by_category = [
        CategorySummaryItem(category=category, total=total)
        for category, total in session.execute(categories_stmt).all()
    ]

    total_expenses = total_expenses or decimal_zero()
    total_income = total_income or decimal_zero()

    return MonthSummaryOutput(
        total_expenses=total_expenses,
        total_income=total_income,
        balance=total_income - total_expenses,
        expenses_by_category=expenses_by_category,
    )


def build_get_month_summary_tool(session_factory: SessionFactory):
    @tool("get_month_summary", args_schema=MonthSummaryInput)
    def get_month_summary_tool(
        user_id: str, month: int, year: int
    ) -> dict[str, object]:
        payload = MonthSummaryInput(user_id=user_id, month=month, year=year)
        with session_factory() as session:
            result = get_month_summary(session, payload)
        return result.model_dump(mode="json")

    return get_month_summary_tool
