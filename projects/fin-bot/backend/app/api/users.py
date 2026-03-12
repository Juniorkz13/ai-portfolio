from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, get_db_session
from app.schemas.user import (
    CurrentUserResponse,
    TelegramLinkCodeResponse,
    UserCreateRequest,
    UserResponse,
    UserTelegramLinkRequest,
)
from app.services.user_service import (
    create_web_user,
    generate_telegram_link_code,
    get_user_by_id,
    get_user_by_telegram_id,
    get_or_create_user_by_telegram,
    link_telegram_to_user,
)

router = APIRouter(tags=["users"])


@router.get("/me", response_model=CurrentUserResponse)
def get_me(current_user=Depends(get_current_user)) -> CurrentUserResponse:
    return CurrentUserResponse.model_validate(current_user)


@router.post("/me/telegram-link-code", response_model=TelegramLinkCodeResponse, status_code=201)
def create_telegram_link_code(
    current_user=Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> TelegramLinkCodeResponse:
    link_code = generate_telegram_link_code(session, user_id=current_user.id)
    return TelegramLinkCodeResponse(code=link_code.code, expires_at=link_code.expires_at)


@router.get("/users/by-telegram/{telegram_id}", response_model=CurrentUserResponse)
def get_user_by_telegram(
    telegram_id: int,
    session: Session = Depends(get_db_session),
) -> CurrentUserResponse:
    user = get_or_create_user_by_telegram(
        session,
        telegram_id=telegram_id,
        name=f"telegram-{telegram_id}",
    )
    return CurrentUserResponse.model_validate(user)


@router.post("/users", response_model=UserResponse, status_code=201)
def create_user_endpoint(
    payload: UserCreateRequest,
    session: Session = Depends(get_db_session),
) -> UserResponse:
    if payload.telegram_id is not None:
        if get_user_by_telegram_id(session, telegram_id=payload.telegram_id) is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Ja existe usuario com este telegram_id.",
            )
        user = create_web_user(session, name=payload.name.strip())
        user = link_telegram_to_user(
            session,
            user_id=user.id,
            telegram_id=payload.telegram_id,
        )
        return UserResponse.model_validate(user)

    user = create_web_user(session, name=payload.name.strip())
    return UserResponse.model_validate(user)


@router.get("/users/{user_id}", response_model=UserResponse)
def get_user_endpoint(
    user_id: uuid.UUID,
    session: Session = Depends(get_db_session),
) -> UserResponse:
    user = get_user_by_id(session, user_id=user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario nao encontrado.")
    return UserResponse.model_validate(user)


@router.patch("/users/{user_id}/telegram", response_model=UserResponse)
def link_user_telegram_endpoint(
    user_id: uuid.UUID,
    payload: UserTelegramLinkRequest,
    session: Session = Depends(get_db_session),
) -> UserResponse:
    user = link_telegram_to_user(
        session,
        user_id=user_id,
        telegram_id=payload.telegram_id,
    )
    return UserResponse.model_validate(user)
