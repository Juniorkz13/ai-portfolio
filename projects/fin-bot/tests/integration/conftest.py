from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api.dependencies import configure_session_factory
from app.api.router import api_router
from bot.telegram.service import TelegramBotService
from bot.telegram.webhook import get_telegram_service, router as telegram_webhook_router


class FakeTelegramBotClient:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str]] = []

    def send_message(self, *, chat_id: int, text: str) -> None:
        self.messages.append((chat_id, text))


@pytest.fixture
def fake_telegram_client() -> FakeTelegramBotClient:
    return FakeTelegramBotClient()


@pytest.fixture
def integration_client(
    session_factory: sessionmaker[Session],
    fake_telegram_client: FakeTelegramBotClient,
) -> TestClient:
    configure_session_factory(session_factory)
    app = FastAPI()
    app.include_router(api_router)
    app.include_router(telegram_webhook_router)
    app.dependency_overrides[get_telegram_service] = lambda: TelegramBotService(  # type: ignore[assignment]
        client=fake_telegram_client
    )
    return TestClient(app)
