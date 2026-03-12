from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import Field

from app.schemas.transaction import SchemaModel


class UserCreateRequest(SchemaModel):
    name: str
    telegram_id: int | None = Field(default=None, ge=1)


class UserTelegramLinkRequest(SchemaModel):
    telegram_id: int = Field(ge=1)


class TelegramLinkCodeResponse(SchemaModel):
    code: str
    expires_at: datetime


class UserResponse(SchemaModel):
    id: uuid.UUID
    name: str
    telegram_id: int | None = Field(default=None, ge=1)
    created_at: datetime


class CurrentUserResponse(UserResponse):
    pass
