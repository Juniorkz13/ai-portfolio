from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.schemas.summary import CategorySummaryResponse, MonthSummaryResponse
from app.schemas.transaction import (
    CsvImportResponse,
    ImportTransactionErrorResponse,
    ImportTransactionRowResponse,
    TransactionCreateRequest,
    TransactionResponse,
)
from app.tools.get_category_summary import get_category_summary
from app.tools.get_month_summary import get_month_summary
from app.tools.import_csv_transactions import import_csv_transactions
from app.tools.list_recent_transactions import list_recent_transactions
from app.tools.save_transaction import save_transaction
from app.tools.types import (
    CsvImportInput,
    GetCategorySummaryInput,
    ListRecentTransactionsInput,
    MonthSummaryInput,
    SaveTransactionInput,
)
from app.services.insights_service import build_month_insights


def save_transaction_service(
    session: Session,
    *,
    user_id: uuid.UUID,
    payload: TransactionCreateRequest,
) -> TransactionResponse:
    result = save_transaction(
        session,
        SaveTransactionInput(
            user_id=user_id,
            type=payload.type,
            amount=payload.amount,
            category=payload.category,
            description=payload.description,
            date=payload.date,
        ),
    )
    return TransactionResponse.model_validate(result.model_dump(by_alias=True))


def list_recent_transactions_service(
    session: Session,
    *,
    user_id: uuid.UUID,
    limit: int,
    month: int | None = None,
    year: int | None = None,
) -> list[TransactionResponse]:
    results = list_recent_transactions(
        session,
        ListRecentTransactionsInput(user_id=user_id, limit=limit),
    )
    items = [
        TransactionResponse.model_validate(result.model_dump(by_alias=True))
        for result in results
    ]

    if month is not None and year is not None:
        items = [item for item in items if item.date.month == month and item.date.year == year]

    return items


def get_month_summary_service(
    session: Session,
    *,
    user_id: uuid.UUID,
    month: int,
    year: int,
) -> MonthSummaryResponse:
    result = get_month_summary(
        session,
        MonthSummaryInput(user_id=user_id, month=month, year=year),
    )
    summary = MonthSummaryResponse.model_validate(result.model_dump())
    summary.insights = build_month_insights(
        summary=summary,
        user_id=user_id,
        month=month,
        year=year,
    )
    return summary


def get_category_summary_service(
    session: Session,
    *,
    user_id: uuid.UUID,
    month: int,
    year: int,
) -> list[CategorySummaryResponse]:
    results = get_category_summary(
        session,
        GetCategorySummaryInput(user_id=user_id, month=month, year=year),
    )
    return [CategorySummaryResponse.model_validate(result.model_dump()) for result in results]


def import_csv_transactions_service(
    session: Session,
    *,
    user_id: uuid.UUID,
    csv_content: str,
) -> CsvImportResponse:
    result = import_csv_transactions(
        session,
        CsvImportInput(user_id=user_id, csv_content=csv_content),
    )
    return CsvImportResponse(
        imported_count=result.imported_count,
        skipped_count=result.skipped_count,
        transactions=[
            ImportTransactionRowResponse(
                line_number=item.line_number,
                transaction=TransactionResponse.model_validate(
                    item.transaction.model_dump(by_alias=True)
                ),
            )
            for item in result.transactions
        ],
        errors=[
            ImportTransactionErrorResponse(
                line_number=item.line_number,
                error=item.error,
            )
            for item in result.errors
        ],
    )
