from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api.dependencies import configure_session_factory
from app.api.router import api_router
from app.core.config import get_settings
from app.db.session import SessionLocal, init_database
from bot.telegram.webhook import router as telegram_webhook_router

settings = get_settings()

app = FastAPI(title=settings.app_name)
app.include_router(api_router)
app.include_router(telegram_webhook_router)
app.mount("/", StaticFiles(directory=PROJECT_ROOT / "frontend", html=True), name="frontend")


@app.on_event("startup")
def on_startup() -> None:
    init_database()
    configure_session_factory(SessionLocal)
