from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Transaction
from app.tools._compat import tool
from app.tools.types import (
    ListRecentTransactionsInput,
    SessionFactory,
    TransactionOutput,
)


def list_recent_transactions(
    session: Session, payload: ListRecentTransactionsInput
) -> list[TransactionOutput]:
    stmt = (
        select(Transaction)
        .where(Transaction.user_id == payload.user_id)
        .order_by(Transaction.date.desc(), Transaction.created_at.desc())
        .limit(payload.limit)
    )
    transactions = session.scalars(stmt).all()

    return [
        TransactionOutput(
            id=transaction.id,
            user_id=transaction.user_id,
            type=transaction.type,
            amount=transaction.amount,
            category=transaction.category,
            description=transaction.description,
            date=transaction.date,
            created_at=transaction.created_at,
            updated_at=transaction.updated_at,
        )
        for transaction in transactions
    ]


def build_list_recent_transactions_tool(session_factory: SessionFactory):
    @tool("list_recent_transactions", args_schema=ListRecentTransactionsInput)
    def list_recent_transactions_tool(
        user_id: str, limit: int = 10
    ) -> list[dict[str, object]]:
        payload = ListRecentTransactionsInput(user_id=user_id, limit=limit)
        with session_factory() as session:
            results = list_recent_transactions(session, payload)
        return [result.model_dump(mode="json", by_alias=True) for result in results]

    return list_recent_transactions_tool
