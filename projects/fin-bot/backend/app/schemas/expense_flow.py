from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.schemas.transaction import TransactionResponse
from app.tools.types import TransactionOutput


class ExpenseMessageRequest(BaseModel):
    message: str


class ExpenseMessageResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    intent: str
    parsed_data: dict[str, object | None]
    saved_transaction: TransactionResponse | None = None
    response_message: str
