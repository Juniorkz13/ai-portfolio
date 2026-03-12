from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db_session
from bot.telegram.client import TelegramBotClient, TelegramClientError
from bot.telegram.config import TelegramBotSettings
from bot.telegram.schemas import TelegramUpdate
from bot.telegram.service import TelegramBotService

router = APIRouter(tags=["telegram"])


def get_telegram_service() -> TelegramBotService:
    settings = TelegramBotSettings()
    return TelegramBotService(client=TelegramBotClient(settings=settings))


@router.post("/webhook/telegram", status_code=status.HTTP_200_OK)
async def telegram_webhook(
    update: TelegramUpdate,
    session: Session = Depends(get_db_session),
    service: TelegramBotService = Depends(get_telegram_service),
) -> dict[str, bool]:
    if update.message is None:
        return {"ok": True}

    try:
        await service.handle_message(session, update.message)
    except TelegramClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return {"ok": True}
