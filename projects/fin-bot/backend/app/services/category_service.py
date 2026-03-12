from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Category, TransactionType
from app.schemas.category import CategoryRead


def list_categories(
    session: Session,
    *,
    transaction_type: TransactionType | None = None,
) -> list[CategoryRead]:
    stmt = select(Category).order_by(Category.name.asc())

    if transaction_type is not None:
        stmt = stmt.where(Category.type == transaction_type)

    categories = session.scalars(stmt).all()
    return [CategoryRead.model_validate(category) for category in categories]
