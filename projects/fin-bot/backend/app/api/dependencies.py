from __future__ import annotations

import uuid
from collections.abc import Generator

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session, sessionmaker

from app.models import User
from app.services.user_service import get_or_create_user_by_telegram, get_user_by_id

_session_factory: sessionmaker[Session] | None = None


def configure_session_factory(factory: sessionmaker[Session]) -> None:
    global _session_factory
    _session_factory = factory


def get_db_session() -> Generator[Session, None, None]:
    if _session_factory is None:
        raise RuntimeError("Session factory nao configurada.")

    session = _session_factory()
    try:
        yield session
    finally:
        session.close()


def get_current_user(
    session: Session = Depends(get_db_session),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    x_telegram_id: str | None = Header(default=None, alias="X-Telegram-Id"),
) -> User:
    return resolve_current_user(
        session,
        x_user_id=x_user_id,
        x_telegram_id=x_telegram_id,
    )


def resolve_current_user(
    session: Session,
    *,
    x_user_id: str | None,
    x_telegram_id: str | None,
) -> User:
    if x_user_id:
        try:
            user_id = uuid.UUID(x_user_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Header X-User-Id invalido.",
            ) from exc

        user = get_user_by_id(session, user_id=user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Usuario nao encontrado.",
            )
        return user

    if x_telegram_id:
        try:
            telegram_id = int(x_telegram_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Header X-Telegram-Id invalido.",
            ) from exc

        return get_or_create_user_by_telegram(
            session,
            telegram_id=telegram_id,
            name=f"telegram-{telegram_id}",
        )

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Informe X-User-Id ou X-Telegram-Id.",
    )


def get_current_user_id(current_user: User = Depends(get_current_user)) -> uuid.UUID:
    return current_user.id
