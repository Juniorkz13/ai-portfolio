from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal
import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.agents.types import IntentType, RouterAgentInput
from app.api.dependencies import configure_session_factory
from app.services.expense_message_flow import ExpenseMessageFlowService
from app.services.user_service import generate_telegram_link_code
from bot.telegram.service import TelegramBotService
from bot.telegram.webhook import get_telegram_service, router as telegram_webhook_router
from bot.telegram.client import TelegramBotClient
from bot.telegram.config import TelegramBotSettings
from bot.telegram.messages import (
    build_recent_transactions_message,
    build_transaction_confirmation,
)
from bot.telegram.schemas import TelegramUpdate
from app.schemas.transaction import TransactionResponse
from app.models.enums import TransactionType
from app.agents.router_agent import RouterAgent
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import AgentLog, Base, Transaction, User


def test_telegram_update_parsing() -> None:
    update = TelegramUpdate.model_validate(
        {
            "update_id": 1,
            "message": {
                "message_id": 10,
                "chat": {"id": 99, "type": "private"},
                "from_user": {"id": 99, "first_name": "Jose"},
                "text": "Gastei 45 no Uber",
            },
        }
    )

    assert update.message is not None
    assert update.message.text == "Gastei 45 no Uber"


def test_router_agent_for_telegram_message() -> None:
    result = RouterAgent().run(RouterAgentInput(message="Quanto gastei esse mes?"))
    assert result.intent == IntentType.GET_SUMMARY


def test_router_agent_for_telegram_insights_message() -> None:
    result = RouterAgent().run(RouterAgentInput(message="Me dê um insight financeiro"))
    assert result.intent == IntentType.GET_INSIGHTS


def test_transaction_confirmation_message() -> None:
    message = build_transaction_confirmation(
        TransactionResponse(
            id="9cf393df-4484-4070-b66c-b1da54455a68",
            user_id="1cf393df-4484-4070-b66c-b1da54455a68",
            type=TransactionType.EXPENSE,
            amount=Decimal("45.00"),
            category="transporte",
            description="Uber",
            date=date(2026, 3, 11),
        )
    )

    assert "transporte" in message


def test_recent_transactions_message() -> None:
    text = build_recent_transactions_message([])
    assert "Nao encontrei" in text


def build_session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    return factory()


def create_user(session: Session) -> User:
    user = User(telegram_id=123456789, name="Jose")
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


class FakeTelegramBotClient:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str]] = []

    def send_message(self, *, chat_id: int, text: str) -> None:
        self.messages.append((chat_id, text))


class StubExpenseMessageService:
    def __init__(self, response_message: str) -> None:
        self.response_message = response_message
        self.calls: list[tuple[str, str]] = []

    def process(self, session: Session, *, user_id, message: str):
        self.calls.append((str(user_id), message))
        return type(
            "StubExpenseResult",
            (),
            {"response_message": self.response_message},
        )()


def test_expense_flow_registers_expense_message() -> None:
    with build_session() as session:
        user = create_user(session)
        service = ExpenseMessageFlowService()

        result = service.process_message(
            session,
            user_id=user.id,
            message="Gastei 35 no uber ontem",
            reference_date=date(2026, 3, 11),
        )

        stored = session.scalar(select(Transaction).where(Transaction.user_id == user.id))
        logs = session.scalars(select(AgentLog)).all()

        assert result.status.value == "completed"
        assert stored is not None
        assert stored.amount == Decimal("35")
        assert stored.category == "transporte"
        assert len(logs) == 3


def test_expense_flow_requests_confirmation_when_category_is_ambiguous() -> None:
    with build_session() as session:
        user = create_user(session)
        service = ExpenseMessageFlowService()

        first_result = service.process_message(
            session,
            user_id=user.id,
            message="Gastei 35 ontem",
            reference_date=date(2026, 3, 11),
        )
        second_result = service.process_message(
            session,
            user_id=user.id,
            message="sim",
            reference_date=date(2026, 3, 11),
        )

        stored = session.scalar(select(Transaction).where(Transaction.user_id == user.id))

        assert first_result.status.value == "confirmation_required"
        assert "Responda 'sim'" in first_result.message
        assert second_result.status.value == "completed"
        assert stored is not None
        assert stored.category == "outros"


def test_telegram_webhook_registers_expense_message() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    configure_session_factory(factory)

    with factory() as session:
        user = create_user(session)
        telegram_id = user.telegram_id

    fake_client = FakeTelegramBotClient()

    app = FastAPI()
    app.include_router(telegram_webhook_router)
    app.dependency_overrides[get_telegram_service] = lambda: TelegramBotService(  # type: ignore[name-defined]
        client=fake_client
    )

    client = TestClient(app)
    response = client.post(
        "/webhook/telegram",
        json={
            "update_id": 1,
            "message": {
                "message_id": 10,
                "chat": {"id": telegram_id, "type": "private"},
                "from": {"id": telegram_id, "first_name": "Jose"},
                "text": "Gastei 45 no Uber ontem",
            },
        },
    )

    with factory() as session:
        stored = session.scalar(select(Transaction).where(Transaction.category == "transporte"))

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert stored is not None
    assert stored.amount == Decimal("45")
    assert fake_client.messages
    assert "Despesa registrada" in fake_client.messages[0][1]


def test_telegram_webhook_rejects_document_import_for_now() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    configure_session_factory(factory)

    with factory() as session:
        user = create_user(session)
        telegram_id = user.telegram_id

    fake_client = FakeTelegramBotClient()

    app = FastAPI()
    app.include_router(telegram_webhook_router)
    app.dependency_overrides[get_telegram_service] = lambda: TelegramBotService(  # type: ignore[name-defined]
        client=fake_client
    )

    client = TestClient(app)
    response = client.post(
        "/webhook/telegram",
        json={
            "update_id": 2,
            "message": {
                "message_id": 11,
                "chat": {"id": telegram_id, "type": "private"},
                "from": {"id": telegram_id, "first_name": "Jose"},
                "document": {"file_id": "abc123", "file_name": "gastos.csv"},
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert fake_client.messages
    assert "apenas mensagem de texto" in fake_client.messages[0][1]


def test_telegram_service_sends_flow_response_to_same_chat_id() -> None:
    with build_session() as session:
        fake_client = FakeTelegramBotClient()
        stub_flow = StubExpenseMessageService("Despesa registrada: R$18.00 em alimentação.")
        service = TelegramBotService(client=fake_client, message_flow_service=stub_flow)  # type: ignore[arg-type]

        update = TelegramUpdate.model_validate(
            {
                "update_id": 3,
                "message": {
                    "message_id": 12,
                    "chat": {"id": 777001, "type": "private"},
                    "from": {"id": 777001, "first_name": "Ana"},
                    "text": "Gastei 18 no almoço hoje",
                },
            }
        )

        assert update.message is not None
        response = asyncio.run(service.handle_message(session, update.message))

        assert response == "Despesa registrada: R$18.00 em alimentação."
        assert fake_client.messages == [
            (777001, "Despesa registrada: R$18.00 em alimentação.")
        ]
        assert stub_flow.calls


def test_telegram_service_logs_chat_id_and_response_message(caplog) -> None:
    with build_session() as session:
        fake_client = FakeTelegramBotClient()
        stub_flow = StubExpenseMessageService("Despesa registrada: R$18.00 em alimentação.")
        service = TelegramBotService(client=fake_client, message_flow_service=stub_flow)  # type: ignore[arg-type]
        update = TelegramUpdate.model_validate(
            {
                "update_id": 4,
                "message": {
                    "message_id": 13,
                    "chat": {"id": 9911, "type": "private"},
                    "from": {"id": 9911, "first_name": "Leo"},
                    "text": "Gastei 18 no almoço hoje",
                },
            }
        )

        caplog.set_level(logging.DEBUG)
        assert update.message is not None
        asyncio.run(service.handle_message(session, update.message))

        assert "Telegram chat_id extracted" in caplog.text
        assert "Telegram response_message generated" in caplog.text


def test_telegram_webhook_registers_income_message() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    configure_session_factory(factory)

    fake_client = FakeTelegramBotClient()

    app = FastAPI()
    app.include_router(telegram_webhook_router)
    app.dependency_overrides[get_telegram_service] = lambda: TelegramBotService(client=fake_client)

    client = TestClient(app)
    response = client.post(
        "/webhook/telegram",
        json={
            "update_id": 5,
            "message": {
                "message_id": 20,
                "chat": {"id": 551122, "type": "private"},
                "from": {"id": 551122, "first_name": "Bia"},
                "text": "Recebi 2500 de salário hoje",
            },
        },
    )

    with factory() as session:
        stored = session.scalar(select(Transaction).where(Transaction.user_id.is_not(None)))

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert stored is not None
    assert stored.type.value == "income"
    assert str(stored.amount) == "2500.00"
    assert stored.category == "salário"
    assert fake_client.messages
    assert "Receita registrada" in fake_client.messages[0][1]


def test_telegram_webhook_returns_month_summary_message() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    configure_session_factory(factory)

    with factory() as session:
        user = create_user(session)
        session.add_all(
            [
                Transaction(
                    user_id=user.id,
                    type=TransactionType.EXPENSE,
                    amount=Decimal("120.00"),
                    category="alimentação",
                    description="almoço",
                    date=date.today(),
                ),
                Transaction(
                    user_id=user.id,
                    type=TransactionType.INCOME,
                    amount=Decimal("500.00"),
                    category="salário",
                    description="salário",
                    date=date.today(),
                ),
            ]
        )
        session.commit()
        telegram_id = user.telegram_id

    fake_client = FakeTelegramBotClient()
    app = FastAPI()
    app.include_router(telegram_webhook_router)
    app.dependency_overrides[get_telegram_service] = lambda: TelegramBotService(client=fake_client)

    client = TestClient(app)
    response = client.post(
        "/webhook/telegram",
        json={
            "update_id": 6,
            "message": {
                "message_id": 21,
                "chat": {"id": telegram_id, "type": "private"},
                "from": {"id": telegram_id, "first_name": "Bia"},
                "text": "Resumo deste mês",
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert fake_client.messages
    assert "Resumo do mes" in fake_client.messages[0][1]
    assert "saldo R$380.00" in fake_client.messages[0][1]
    assert "Insights:" in fake_client.messages[0][1]
    assert "alimentação" in fake_client.messages[0][1].lower()


def test_telegram_webhook_returns_recent_history_message() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    configure_session_factory(factory)

    with factory() as session:
        user = create_user(session)
        session.add_all(
            [
                Transaction(
                    user_id=user.id,
                    type=TransactionType.EXPENSE,
                    amount=Decimal("20.00"),
                    category="alimentação",
                    description="almoço",
                    date=date.today(),
                ),
                Transaction(
                    user_id=user.id,
                    type=TransactionType.INCOME,
                    amount=Decimal("300.00"),
                    category="salário",
                    description="salário",
                    date=date.today(),
                ),
            ]
        )
        session.commit()
        telegram_id = user.telegram_id

    fake_client = FakeTelegramBotClient()
    app = FastAPI()
    app.include_router(telegram_webhook_router)
    app.dependency_overrides[get_telegram_service] = lambda: TelegramBotService(client=fake_client)

    client = TestClient(app)
    response = client.post(
        "/webhook/telegram",
        json={
            "update_id": 7,
            "message": {
                "message_id": 22,
                "chat": {"id": telegram_id, "type": "private"},
                "from": {"id": telegram_id, "first_name": "Bia"},
                "text": "Mostre minhas últimas transações",
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert fake_client.messages
    assert "Suas transacoes mais recentes" in fake_client.messages[0][1]
    assert "expense" in fake_client.messages[0][1]


def test_telegram_webhook_returns_month_insights_message() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    configure_session_factory(factory)

    with factory() as session:
        user = create_user(session)
        session.add_all(
            [
                Transaction(
                    user_id=user.id,
                    type=TransactionType.EXPENSE,
                    amount=Decimal("220.00"),
                    category="alimentação",
                    description="restaurante",
                    date=date.today(),
                ),
                Transaction(
                    user_id=user.id,
                    type=TransactionType.INCOME,
                    amount=Decimal("1000.00"),
                    category="salário",
                    description="salário",
                    date=date.today(),
                ),
            ]
        )
        session.commit()
        telegram_id = user.telegram_id

    fake_client = FakeTelegramBotClient()
    app = FastAPI()
    app.include_router(telegram_webhook_router)
    app.dependency_overrides[get_telegram_service] = lambda: TelegramBotService(client=fake_client)

    client = TestClient(app)
    response = client.post(
        "/webhook/telegram",
        json={
            "update_id": 8,
            "message": {
                "message_id": 23,
                "chat": {"id": telegram_id, "type": "private"},
                "from": {"id": telegram_id, "first_name": "Bia"},
                "text": "Analise minhas finanças deste mês",
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert fake_client.messages
    assert "Insights financeiros" in fake_client.messages[0][1]
    assert "alimentação" in fake_client.messages[0][1].lower()


def test_telegram_webhook_links_web_user_with_code() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    configure_session_factory(factory)

    with factory() as session:
        user = User(name="Usuario Web", telegram_id=None)
        session.add(user)
        session.commit()
        session.refresh(user)
        link_code = generate_telegram_link_code(session, user_id=user.id)

    fake_client = FakeTelegramBotClient()
    app = FastAPI()
    app.include_router(telegram_webhook_router)
    app.dependency_overrides[get_telegram_service] = lambda: TelegramBotService(client=fake_client)

    client = TestClient(app)
    response = client.post(
        "/webhook/telegram",
        json={
            "update_id": 9,
            "message": {
                "message_id": 24,
                "chat": {"id": 5808218273, "type": "private"},
                "from": {"id": 5808218273, "first_name": "Bia"},
                "text": f"/link {link_code.code}",
            },
        },
    )

    with factory() as session:
        linked_user = session.get(User, user.id)

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert linked_user is not None
    assert linked_user.telegram_id == 5808218273
    assert fake_client.messages
    assert "Conta vinculada com sucesso" in fake_client.messages[0][1]


def test_telegram_webhook_rejects_reused_link_code() -> None:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    configure_session_factory(factory)

    with factory() as session:
        user = User(name="Usuario Web", telegram_id=None)
        session.add(user)
        session.commit()
        session.refresh(user)
        link_code = generate_telegram_link_code(session, user_id=user.id)

    fake_client = FakeTelegramBotClient()
    app = FastAPI()
    app.include_router(telegram_webhook_router)
    app.dependency_overrides[get_telegram_service] = lambda: TelegramBotService(client=fake_client)

    client = TestClient(app)
    first_response = client.post(
        "/webhook/telegram",
        json={
            "update_id": 10,
            "message": {
                "message_id": 25,
                "chat": {"id": 700001, "type": "private"},
                "from": {"id": 700001, "first_name": "Ana"},
                "text": f"/link {link_code.code}",
            },
        },
    )
    second_response = client.post(
        "/webhook/telegram",
        json={
            "update_id": 11,
            "message": {
                "message_id": 26,
                "chat": {"id": 700001, "type": "private"},
                "from": {"id": 700001, "first_name": "Ana"},
                "text": f"/link {link_code.code}",
            },
        },
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert fake_client.messages
    assert "Conta vinculada com sucesso" in fake_client.messages[0][1]
    assert "Codigo de vinculo invalido ou expirado." in fake_client.messages[1][1]


def test_telegram_client_logs_send_message_response(monkeypatch, caplog) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def getcode(self) -> int:
            return 200

        def read(self) -> bytes:
            return b'{\"ok\":true,\"result\":{\"message_id\":10}}'

    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout=15):
        captured["url"] = request.full_url
        captured["data"] = request.data
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("bot.telegram.client.urlopen", fake_urlopen)
    client = TelegramBotClient(settings=TelegramBotSettings(token="test-token"))

    caplog.set_level(logging.DEBUG)
    client.send_message(chat_id=12345, text="oi")

    assert captured["url"] == "https://api.telegram.org/bottest-token/sendMessage"
    assert "Telegram sendMessage response received" in caplog.text
    assert "message_id" in caplog.text
