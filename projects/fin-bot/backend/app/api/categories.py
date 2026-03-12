from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.dependencies import get_db_session
from app.models import TransactionType
from app.schemas.category import CategoryRead
from app.services.category_service import list_categories

router = APIRouter(prefix="/categories", tags=["categories"])


@router.get("", response_model=list[CategoryRead])
def get_categories(
    transaction_type: TransactionType | None = Query(default=None, alias="type"),
    session: Session = Depends(get_db_session),
) -> list[CategoryRead]:
    return list_categories(session, transaction_type=transaction_type)
