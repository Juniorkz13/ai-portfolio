from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, get_db_session
from app.models import User
from app.schemas.summary import MonthSummaryResponse
from app.services.transaction_service import get_month_summary_service

router = APIRouter(prefix="/summary", tags=["summary"])


@router.get("/month", response_model=MonthSummaryResponse)
def get_month_summary_endpoint(
    month: int = Query(..., ge=1, le=12),
    year: int = Query(..., ge=2000, le=3000),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> MonthSummaryResponse:
    return get_month_summary_service(session, user_id=current_user.id, month=month, year=year)
