from __future__ import annotations

import json
from dataclasses import dataclass
import logging
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from bot.telegram.config import TelegramBotSettings

logger = logging.getLogger(__name__)


class TelegramClientError(RuntimeError):
    pass


@dataclass(slots=True)
class TelegramBotClient:
    settings: TelegramBotSettings

    def send_message(self, *, chat_id: int, text: str) -> None:
        logger.warning(
            "Telegram send_message called enabled=%s chat_id=%s text=%s",
            self.settings.enabled,
            chat_id,
            text,
        )

        if not self.settings.enabled:
            logger.warning("Telegram disabled: message not sent")
            return

        payload = urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
        request = Request(
            url=f"{self.settings.bot_api_url}/sendMessage",
            data=payload,
            method="POST",
        )
        status_code, response_body = self._open(request)
        logger.warning(
            "Telegram sendMessage response received chat_id=%s status_code=%s body=%s",
            chat_id,
            status_code,
            response_body.decode("utf-8", errors="replace"),
        )

    def download_file_content(self, file_id: str) -> str:
        if not self.settings.enabled:
            raise TelegramClientError("Telegram bot token nao configurado.")

        request = Request(f"{self.settings.bot_api_url}/getFile?file_id={file_id}", method="GET")
        _, response = self._open(request)
        payload = json.loads(response.decode("utf-8"))
        file_path = payload["result"]["file_path"]

        file_request = Request(f"{self.settings.file_api_url}/{file_path}", method="GET")
        _, file_content = self._open(file_request)
        return file_content.decode("utf-8")

    def _open(self, request: Request) -> tuple[int, bytes]:
        try:
            with urlopen(request, timeout=15) as response:
                return response.getcode(), response.read()
        except (HTTPError, URLError) as exc:
            raise TelegramClientError("Falha ao comunicar com a API do Telegram.") from exc