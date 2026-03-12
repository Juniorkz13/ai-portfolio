from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TelegramSchema(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class TelegramUser(TelegramSchema):
    id: int
    first_name: str | None = None
    username: str | None = None


class TelegramChat(TelegramSchema):
    id: int
    type: str | None = None


class TelegramDocument(TelegramSchema):
    file_id: str
    file_name: str | None = None
    mime_type: str | None = None


class TelegramMessage(TelegramSchema):
    message_id: int
    chat: TelegramChat
    from_user: TelegramUser | None = Field(default=None, alias="from")
    text: str | None = None
    caption: str | None = None
    document: TelegramDocument | None = None


class TelegramUpdate(TelegramSchema):
    update_id: int
    message: TelegramMessage | None = None
