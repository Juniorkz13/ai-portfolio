from __future__ import annotations

from calendar import monthrange
from datetime import date
from decimal import Decimal


def get_month_date_range(year: int, month: int) -> tuple[date, date]:
    start_date = date(year, month, 1)
    end_date = date(year, month, monthrange(year, month)[1])
    return start_date, end_date


def decimal_zero() -> Decimal:
    return Decimal("0.00")
