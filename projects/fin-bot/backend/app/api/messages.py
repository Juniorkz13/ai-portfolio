from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, get_db_session
from app.models import User
from app.schemas.expense_flow import ExpenseMessageRequest, ExpenseMessageResponse
from app.services.expense_message_service import ExpenseMessageService

router = APIRouter(prefix="/messages", tags=["messages"])


@router.post("/transaction", response_model=ExpenseMessageResponse)
@router.post("/expense", response_model=ExpenseMessageResponse, include_in_schema=False)
def process_message_transaction(
    payload: ExpenseMessageRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> ExpenseMessageResponse:
    service = ExpenseMessageService()
    return service.process(session, user_id=current_user.id, message=payload.message)
