from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.models import Transaction, User
from app.models.enums import TransactionType


def test_web_user_creation_and_later_telegram_link(
    integration_client: TestClient,
) -> None:
    create_response = integration_client.post("/api/users", json={"name": "Maria Web"})
    assert create_response.status_code == 201
    created_user = create_response.json()
    assert created_user["telegram_id"] is None

    link_response = integration_client.patch(
        f"/api/users/{created_user['id']}/telegram",
        json={"telegram_id": 5808218273},
    )
    assert link_response.status_code == 200
    assert link_response.json()["telegram_id"] == 5808218273


def test_duplicate_telegram_id_is_rejected(
    integration_client: TestClient,
) -> None:
    first = integration_client.post("/api/users", json={"name": "Primeiro"}).json()
    second = integration_client.post("/api/users", json={"name": "Segundo"}).json()

    first_link = integration_client.patch(
        f"/api/users/{first['id']}/telegram",
        json={"telegram_id": 5808218273},
    )
    second_link = integration_client.patch(
        f"/api/users/{second['id']}/telegram",
        json={"telegram_id": 5808218273},
    )

    assert first_link.status_code == 200
    assert second_link.status_code == 409


def test_transactions_summary_history_and_insights_are_isolated_per_user(
    integration_client: TestClient,
    session_factory: sessionmaker[Session],
) -> None:
    first_user = integration_client.post("/api/users", json={"name": "Ana"}).json()
    second_user = integration_client.post("/api/users", json={"name": "Bia"}).json()

    first_headers = {"X-User-Id": first_user["id"]}
    second_headers = {"X-User-Id": second_user["id"]}

    first_expense = integration_client.post(
        "/api/transactions",
        headers=first_headers,
        json={
            "type": "expense",
            "amount": "35.00",
            "category": "alimentação",
            "description": "almoço",
            "date": str(date.today()),
        },
    )
    second_income = integration_client.post(
        "/api/transactions",
        headers=second_headers,
        json={
            "type": "income",
            "amount": "1500.00",
            "category": "salário",
            "description": "salário",
            "date": str(date.today()),
        },
    )
    assert first_expense.status_code == 201
    assert second_income.status_code == 201

    with session_factory() as session:
        session.add(
            Transaction(
                user_id=uuid.UUID(second_user["id"]),
                type=TransactionType.EXPENSE,
                amount=Decimal("400.00"),
                category="moradia",
                description="aluguel",
                date=date.today(),
            )
        )
        session.commit()

    first_transactions = integration_client.get("/api/transactions?limit=10", headers=first_headers)
    second_transactions = integration_client.get("/api/transactions?limit=10", headers=second_headers)
    assert first_transactions.status_code == 200
    assert second_transactions.status_code == 200
    assert len(first_transactions.json()["items"]) == 1
    assert first_transactions.json()["items"][0]["description"] == "almoço"
    assert len(second_transactions.json()["items"]) == 2

    first_summary = integration_client.get(
        f"/api/summary/month?month={date.today().month}&year={date.today().year}",
        headers=first_headers,
    )
    second_summary = integration_client.get(
        f"/api/summary/month?month={date.today().month}&year={date.today().year}",
        headers=second_headers,
    )
    assert first_summary.status_code == 200
    assert second_summary.status_code == 200
    assert first_summary.json()["total_expenses"] == "35.00"
    assert second_summary.json()["total_income"] == "1500.00"
    assert second_summary.json()["total_expenses"] == "400.00"

    first_history = integration_client.post(
        "/api/messages/transaction",
        headers=first_headers,
        json={"message": "Mostre minhas últimas transações"},
    )
    second_history = integration_client.post(
        "/api/messages/transaction",
        headers=second_headers,
        json={"message": "Mostre minhas últimas transações"},
    )
    assert first_history.status_code == 200
    assert second_history.status_code == 200
    assert len(first_history.json()["parsed_data"]["items"]) == 1
    assert first_history.json()["parsed_data"]["items"][0]["description"] == "almoço"
    assert len(second_history.json()["parsed_data"]["items"]) == 2

    first_insights = integration_client.post(
        "/api/messages/transaction",
        headers=first_headers,
        json={"message": "Analise minhas finanças deste mês"},
    )
    second_insights = integration_client.post(
        "/api/messages/transaction",
        headers=second_headers,
        json={"message": "Analise minhas finanças deste mês"},
    )
    assert first_insights.status_code == 200
    assert second_insights.status_code == 200
    assert "alimentação" in first_insights.json()["response_message"].lower()
    assert "moradia" not in first_insights.json()["response_message"].lower()
    assert "moradia" in second_insights.json()["response_message"].lower()


def test_csv_import_is_associated_with_the_correct_user(
    integration_client: TestClient,
    session_factory: sessionmaker[Session],
) -> None:
    first_user = integration_client.post("/api/users", json={"name": "Csv A"}).json()
    second_user = integration_client.post("/api/users", json={"name": "Csv B"}).json()

    csv_content = "\n".join(
        [
            "type,amount,category,description,date",
            "expense,15.50,transporte,Onibus,2026-03-01",
            "income,300.00,freelance,Projeto,2026-03-02",
        ]
    )

    response = integration_client.post(
        "/api/transactions/import",
        headers={"X-User-Id": first_user["id"]},
        files={"file": ("import.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 201
    assert response.json()["imported_count"] == 2

    with session_factory() as session:
        first_count = (
            session.query(Transaction)
            .filter(Transaction.user_id == uuid.UUID(first_user["id"]))
            .count()
        )
        second_count = (
            session.query(Transaction)
            .filter(Transaction.user_id == uuid.UUID(second_user["id"]))
            .count()
        )

    assert first_count == 2
    assert second_count == 0
