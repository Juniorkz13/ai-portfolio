from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, get_db_session
from app.models import User
from app.schemas.transaction import (
    CsvImportResponse,
    TransactionCreateRequest,
    TransactionListResponse,
    TransactionResponse,
)
from app.services.transaction_service import (
    import_csv_transactions_service,
    list_recent_transactions_service,
    save_transaction_service,
)

router = APIRouter(prefix="/transactions", tags=["transactions"])


@router.post("", response_model=TransactionResponse, status_code=status.HTTP_201_CREATED)
def create_transaction(
    payload: TransactionCreateRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> TransactionResponse:
    return save_transaction_service(session, user_id=current_user.id, payload=payload)


@router.get("", response_model=TransactionListResponse)
def get_transactions(
    limit: int = Query(default=10, ge=1, le=100),
    month: int | None = Query(default=None, ge=1, le=12),
    year: int | None = Query(default=None, ge=2000, le=3000),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> TransactionListResponse:
    if (month is None) != (year is None):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="month e year devem ser informados juntos.",
        )

    items = list_recent_transactions_service(
        session,
        user_id=current_user.id,
        limit=limit,
        month=month,
        year=year,
    )
    return TransactionListResponse(items=items)


@router.post("/import", response_model=CsvImportResponse, status_code=status.HTTP_201_CREATED)
async def import_transactions_csv(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> CsvImportResponse:
    content = await file.read()

    try:
        csv_content = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Arquivo CSV deve estar em UTF-8.",
        ) from exc

    return import_csv_transactions_service(session, user_id=current_user.id, csv_content=csv_content)
