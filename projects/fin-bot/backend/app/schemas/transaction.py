from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import TransactionType


class SchemaModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, from_attributes=True)


class TransactionCreateRequest(SchemaModel):
    type: TransactionType
    amount: Decimal
    category: str
    description: str | None = None
    date: date


class TransactionResponse(SchemaModel):
    id: uuid.UUID
    user_id: uuid.UUID
    type: TransactionType
    amount: Decimal
    category: str
    description: str | None = None
    date: date
    created_at: datetime | None = None
    updated_at: datetime | None = None


class TransactionListResponse(SchemaModel):
    items: list[TransactionResponse]


class ImportTransactionRowResponse(SchemaModel):
    line_number: int
    transaction: TransactionResponse


class ImportTransactionErrorResponse(SchemaModel):
    line_number: int
    error: str


class CsvImportResponse(SchemaModel):
    imported_count: int
    skipped_count: int = 0
    transactions: list[ImportTransactionRowResponse]
    errors: list[ImportTransactionErrorResponse] = Field(default_factory=list)
