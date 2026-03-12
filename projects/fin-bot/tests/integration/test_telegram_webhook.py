from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.models import User


def test_telegram_webhook_handles_start_and_help(
    integration_client: TestClient,
    fake_telegram_client,
) -> None:
    start_response = integration_client.post(
        "/webhook/telegram",
        json={
            "update_id": 1,
            "message": {
                "message_id": 10,
                "chat": {"id": 900001, "type": "private"},
                "from": {"id": 900001, "first_name": "Ana"},
                "text": "/start",
            },
        },
    )
    help_response = integration_client.post(
        "/webhook/telegram",
        json={
            "update_id": 2,
            "message": {
                "message_id": 11,
                "chat": {"id": 900001, "type": "private"},
                "from": {"id": 900001, "first_name": "Ana"},
                "text": "/help",
            },
        },
    )

    assert start_response.status_code == 200
    assert help_response.status_code == 200
    assert "assistente financeiro" in fake_telegram_client.messages[0][1].lower()
    assert "/link CODIGO" in fake_telegram_client.messages[1][1]


def test_telegram_webhook_registers_expense_and_income_for_isolated_users(
    integration_client: TestClient,
    fake_telegram_client,
    session_factory: sessionmaker[Session],
) -> None:
    expense_response = integration_client.post(
        "/webhook/telegram",
        json={
            "update_id": 3,
            "message": {
                "message_id": 12,
                "chat": {"id": 700001, "type": "private"},
                "from": {"id": 700001, "first_name": "Ana"},
                "text": "Gastei 45 no Uber hoje",
            },
        },
    )
    income_response = integration_client.post(
        "/webhook/telegram",
        json={
            "update_id": 4,
            "message": {
                "message_id": 13,
                "chat": {"id": 800001, "type": "private"},
                "from": {"id": 800001, "first_name": "Bia"},
                "text": "Recebi 2500 de salário hoje",
            },
        },
    )

    assert expense_response.status_code == 200
    assert income_response.status_code == 200
    assert "Despesa registrada" in fake_telegram_client.messages[0][1]
    assert "Receita registrada" in fake_telegram_client.messages[1][1]

    with session_factory() as session:
        first_user = session.query(User).filter(User.telegram_id == 700001).one()
        second_user = session.query(User).filter(User.telegram_id == 800001).one()
        assert first_user.id != second_user.id
        assert len(first_user.transactions) == 1
        assert len(second_user.transactions) == 1
        assert first_user.transactions[0].type.value == "expense"
        assert second_user.transactions[0].type.value == "income"
