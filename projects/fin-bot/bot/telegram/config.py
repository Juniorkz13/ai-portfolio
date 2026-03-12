from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class TelegramBotSettings:
    token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    api_base_url: str = os.getenv("TELEGRAM_API_BASE_URL", "https://api.telegram.org")

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    @property
    def bot_api_url(self) -> str:
        return f"{self.api_base_url}/bot{self.token}"

    @property
    def file_api_url(self) -> str:
        return f"{self.api_base_url}/file/bot{self.token}"
