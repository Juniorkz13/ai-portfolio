from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class SchemaModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class CategorySummaryResponse(SchemaModel):
    category: str
    total: Decimal


class MonthSummaryResponse(SchemaModel):
    total_expenses: Decimal
    total_income: Decimal
    balance: Decimal
    expenses_by_category: list[CategorySummaryResponse]
    insights: list[str] = Field(default_factory=list)
