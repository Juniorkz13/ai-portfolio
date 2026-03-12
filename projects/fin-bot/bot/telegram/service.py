from __future__ import annotations

from dataclasses import dataclass, field
import logging
import uuid

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.services.expense_message_service import ExpenseMessageService
from app.services.user_service import consume_telegram_link_code, get_or_create_user_by_telegram
from bot.telegram.client import TelegramBotClient
from bot.telegram.schemas import TelegramMessage

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TelegramBotService:
    client: TelegramBotClient
    message_flow_service: ExpenseMessageService = field(default_factory=ExpenseMessageService)

    async def handle_message(self, session: Session, message: TelegramMessage) -> str:
        chat_id = message.chat.id
        logger.debug("Telegram chat_id extracted chat_id=%s", chat_id)
        sender_telegram_id = message.from_user.id if message.from_user else chat_id
        raw_text = (message.text or message.caption or "").strip()

        if raw_text.lower().startswith("/link"):
            response_text = self._handle_link_command(
                session,
                telegram_id=sender_telegram_id,
                raw_text=raw_text,
            )
        else:
            user = get_or_create_user_by_telegram(
                session,
                telegram_id=sender_telegram_id,
                name=(message.from_user.first_name if message.from_user else None) or "Telegram User",
            )

            if message.document is not None:
                response_text = "No momento o bot aceita apenas mensagem de texto para registrar transações."
            else:
                response_text = self._handle_text(session, user.id, raw_text)

        logger.debug(
            "Telegram response_message generated chat_id=%s response_message=%s",
            chat_id,
            response_text,
        )
        self.client.send_message(chat_id=chat_id, text=response_text)
        return response_text

    def _handle_text(self, session: Session, user_id: uuid.UUID, raw_text: str) -> str:
        if not raw_text:
            return "Envie uma mensagem com a transação que deseja registrar."

        normalized_text = raw_text.strip().lower()

        if normalized_text == "/start":
            return (
                "Olá! Eu sou seu assistente financeiro.\n\n"
                "Você pode me enviar mensagens como:\n"
                "- Gastei 35 no almoço hoje\n"
                "- Recebi 2500 de salário hoje\n"
                "- Quanto gastei esse mês?\n"
                "- Quais foram meus últimos gastos?\n\n"
                "Se quiser vincular sua conta web, gere um código no dashboard e envie /link CODIGO.\n"
                "Use /help para ver mais exemplos."
            )

        if normalized_text == "/help":
            return (
                "Aqui estão alguns exemplos do que você pode me pedir:\n\n"
                "Registrar despesa:\n"
                "• Gastei 35 no almoço hoje\n"
                "• Gastei 18 no Uber ontem\n\n"
                "Registrar receita:\n"
                "• Recebi 2500 de salário hoje\n"
                "• Recebi 800 de freela ontem\n\n"
                "Consultar resumo:\n"
                "• Quanto gastei esse mês?\n"
                "• Qual meu saldo do mês?\n\n"
                "Consultar histórico:\n"
                "• Quais foram meus últimos gastos?\n"
                "• Mostre minhas últimas transações\n\n"
                "Vincular conta web:\n"
                "• /link CODIGO"
            )

        result = self.message_flow_service.process(
            session,
            user_id=user_id,
            message=raw_text,
        )
        return result.response_message

    def _handle_link_command(self, session: Session, *, telegram_id: int, raw_text: str) -> str:
        parts = raw_text.strip().split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            return "Envie o comando no formato /link CODIGO."

        try:
            user = consume_telegram_link_code(
                session,
                code=parts[1],
                telegram_id=telegram_id,
            )
        except HTTPException as exc:
            return exc.detail

        return f"Conta vinculada com sucesso ao usuário {user.name}."
