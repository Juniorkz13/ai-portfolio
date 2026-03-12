from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import Transaction
from app.tools._compat import tool
from app.tools.types import SaveTransactionInput, SessionFactory, TransactionOutput


def save_transaction(
    session: Session,
    payload: SaveTransactionInput,
    *,
    commit: bool = True,
) -> TransactionOutput:
    transaction = Transaction(
        user_id=payload.user_id,
        type=payload.transaction_type,
        amount=payload.amount,
        category=payload.category,
        description=payload.description,
        date=payload.date,
    )
    session.add(transaction)
    session.flush()
    session.refresh(transaction)

    if commit:
        session.commit()

    return TransactionOutput(
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


def build_save_transaction_tool(session_factory: SessionFactory):
    @tool("save_transaction", args_schema=SaveTransactionInput)
    def save_transaction_tool(
        user_id: str,
        transaction_type: str,
        amount: str,
        category: str,
        description: str | None = None,
        date: str | None = None,
    ) -> dict[str, object]:
        payload = SaveTransactionInput(
            user_id=user_id,
            type=transaction_type,
            amount=amount,
            category=category,
            description=description,
            date=date,
        )
        with session_factory() as session:
            result = save_transaction(session, payload)
        return result.model_dump(mode="json", by_alias=True)

    return save_transaction_tool
