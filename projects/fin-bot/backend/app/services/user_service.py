from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
import secrets

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import TelegramLinkCode, User


def get_user_by_id(session: Session, *, user_id: uuid.UUID) -> User | None:
    stmt = select(User).where(User.id == user_id)
    return session.scalar(stmt)


def get_user_by_telegram_id(session: Session, *, telegram_id: int) -> User | None:
    stmt = select(User).where(User.telegram_id == telegram_id)
    return session.scalar(stmt)


def create_web_user(
    session: Session,
    *,
    name: str,
) -> User:
    user = User(name=name, telegram_id=None)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def link_telegram_to_user(
    session: Session,
    *,
    user_id: uuid.UUID,
    telegram_id: int,
) -> User:
    user = get_user_by_id(session, user_id=user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuario nao encontrado.",
        )

    if user.telegram_id is not None and user.telegram_id != telegram_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Usuario ja vinculado a outro telegram_id.",
        )

    existing_user = get_user_by_telegram_id(session, telegram_id=telegram_id)
    if existing_user is not None and existing_user.id != user.id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ja existe usuario com este telegram_id.",
        )

    user.telegram_id = telegram_id
    session.commit()
    session.refresh(user)
    return user


def generate_telegram_link_code(
    session: Session,
    *,
    user_id: uuid.UUID,
    ttl_minutes: int = 10,
) -> TelegramLinkCode:
    user = get_user_by_id(session, user_id=user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usuario nao encontrado.",
        )

    now = datetime.now(UTC)
    active_codes = session.scalars(
        select(TelegramLinkCode).where(
            TelegramLinkCode.user_id == user_id,
            TelegramLinkCode.used_at.is_(None),
            TelegramLinkCode.expires_at > now,
        )
    ).all()
    for item in active_codes:
        item.used_at = now

    code = _generate_code()
    while session.scalar(select(TelegramLinkCode).where(TelegramLinkCode.code == code)) is not None:
        code = _generate_code()

    link_code = TelegramLinkCode(
        user_id=user_id,
        code=code,
        expires_at=now + timedelta(minutes=ttl_minutes),
    )
    session.add(link_code)
    session.commit()
    session.refresh(link_code)
    return link_code


def consume_telegram_link_code(
    session: Session,
    *,
    code: str,
    telegram_id: int,
) -> User:
    now = datetime.now(UTC)
    normalized_code = code.strip().upper()
    link_code = session.scalar(
        select(TelegramLinkCode).where(
            TelegramLinkCode.code == normalized_code,
            TelegramLinkCode.used_at.is_(None),
            TelegramLinkCode.expires_at > now,
        )
    )
    if link_code is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Codigo de vinculo invalido ou expirado.",
        )

    user = link_telegram_to_user(
        session,
        user_id=link_code.user_id,
        telegram_id=telegram_id,
    )
    link_code.used_at = now
    session.commit()
    session.refresh(user)
    return user


def _generate_code() -> str:
    return secrets.token_hex(3).upper()


def get_or_create_user_by_telegram(
    session: Session,
    *,
    telegram_id: int,
    name: str,
) -> User:
    user = get_user_by_telegram_id(session, telegram_id=telegram_id)

    if user is not None:
        if user.name != name and name:
            user.name = name
            session.commit()
            session.refresh(user)
        return user

    user = User(telegram_id=telegram_id, name=name or f"telegram-{telegram_id}")
    session.add(user)
    session.commit()
    session.refresh(user)
    return user
