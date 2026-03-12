from __future__ import annotations

import csv
import uuid
from collections.abc import Callable, Iterable
from datetime import date, datetime
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.orm import Session

from app.models.enums import TransactionType

SessionFactory: TypeAlias = Callable[[], Session]


class ToolModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=False,
        arbitrary_types_allowed=True,
    )


class SaveTransactionInput(ToolModel):
    user_id: uuid.UUID
    transaction_type: TransactionType = Field(alias="type")
    amount: Decimal
    category: str
    description: str | None = None
    date: date


class TransactionOutput(ToolModel):
    id: uuid.UUID
    user_id: uuid.UUID
    transaction_type: TransactionType = Field(alias="type")
    amount: Decimal
    category: str
    description: str | None = None
    date: date
    created_at: datetime | None = None
    updated_at: datetime | None = None


class MonthSummaryInput(ToolModel):
    user_id: uuid.UUID
    month: int = Field(ge=1, le=12)
    year: int = Field(ge=2000, le=3000)


class CategorySummaryItem(ToolModel):
    category: str
    total: Decimal


class MonthSummaryOutput(ToolModel):
    total_expenses: Decimal
    total_income: Decimal
    balance: Decimal
    expenses_by_category: list[CategorySummaryItem]


class ListRecentTransactionsInput(ToolModel):
    user_id: uuid.UUID
    limit: int = Field(default=10, ge=1, le=100)


class CsvImportInput(ToolModel):
    user_id: uuid.UUID
    csv_content: str | None = None
    file_path: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "CsvImportInput":
        if bool(self.csv_content) == bool(self.file_path):
            raise ValueError("Informe exatamente um entre csv_content e file_path.")
        return self


class ImportedTransactionRow(ToolModel):
    line_number: int
    transaction: TransactionOutput


class ImportCsvErrorRow(ToolModel):
    line_number: int
    error: str


class ImportCsvTransactionsOutput(ToolModel):
    imported_count: int
    skipped_count: int = 0
    transactions: list[ImportedTransactionRow]
    errors: list[ImportCsvErrorRow] = Field(default_factory=list)


class GetCategorySummaryInput(ToolModel):
    user_id: uuid.UUID
    month: int = Field(ge=1, le=12)
    year: int = Field(ge=2000, le=3000)


def load_csv_rows(payload: CsvImportInput) -> Iterable[dict[str, str]]:
    if payload.csv_content is not None:
        content = payload.csv_content
    else:
        content = Path(payload.file_path or "").read_text(encoding="utf-8")

    return csv.DictReader(StringIO(content))
