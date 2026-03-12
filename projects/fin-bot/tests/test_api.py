from __future__ import annotations

from datetime import date
from decimal import Decimal

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.models import TransactionType, User
from app.models.transaction import Transaction

def test_create_transaction_endpoint(
    client: TestClient,
    user: User,
) -> None:
    response = client.post(
        "/api/transactions",
        headers={"X-User-Id": str(user.id)},
        json={
            "type": "expense",
            "amount": "19.90",
            "category": "transporte",
            "description": "Metro",
            "date": "2026-03-11",
        },
    )

    assert response.status_code == 201
    assert response.json()["category"] == "transporte"
    assert response.json()["type"] == "expense"


def test_get_me_endpoint_with_x_user_id(
    client: TestClient,
    user: User,
) -> None:
    response = client.get(
        "/api/me",
        headers={"X-User-Id": str(user.id)},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(user.id)
    assert body["telegram_id"] == user.telegram_id
    assert body["created_at"]


def test_get_me_endpoint_with_x_telegram_id_creates_user(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/me",
        headers={"X-Telegram-Id": "998877"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["telegram_id"] == 998877
    assert body["name"] == "telegram-998877"
    assert body["created_at"]


def test_get_user_by_telegram_endpoint_creates_user(
    client: TestClient,
) -> None:
    response = client.get("/api/users/by-telegram/445566")

    assert response.status_code == 200
    body = response.json()
    assert body["telegram_id"] == 445566
    assert body["name"] == "telegram-445566"
    assert body["created_at"]


def test_create_user_endpoint_creates_manual_user(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/users",
        json={"name": "Maria Web"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Maria Web"
    assert body["telegram_id"] is None
    assert body["created_at"]


def test_create_user_endpoint_with_telegram_id_creates_telegram_user(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/users",
        json={"name": "Maria Telegram", "telegram_id": 5808218273},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Maria Telegram"
    assert body["telegram_id"] == 5808218273


def test_get_user_by_id_endpoint_returns_user(
    client: TestClient,
    user: User,
) -> None:
    response = client.get(f"/api/users/{user.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(user.id)
    assert body["name"] == user.name


def test_link_telegram_to_existing_web_user(
    client: TestClient,
) -> None:
    create_response = client.post("/api/users", json={"name": "Usuario Web"})
    user_id = create_response.json()["id"]

    response = client.patch(
        f"/api/users/{user_id}/telegram",
        json={"telegram_id": 5808218273},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == user_id
    assert body["telegram_id"] == 5808218273


def test_link_telegram_rejects_duplicate_telegram_id(
    client: TestClient,
) -> None:
    first_user = client.post("/api/users", json={"name": "Primeiro"}).json()
    second_user = client.post("/api/users", json={"name": "Segundo"}).json()

    first_link = client.patch(
        f"/api/users/{first_user['id']}/telegram",
        json={"telegram_id": 5808218273},
    )
    duplicate_link = client.patch(
        f"/api/users/{second_user['id']}/telegram",
        json={"telegram_id": 5808218273},
    )

    assert first_link.status_code == 200
    assert duplicate_link.status_code == 409
    assert duplicate_link.json()["detail"] == "Ja existe usuario com este telegram_id."


def test_generate_telegram_link_code_for_current_user(
    client: TestClient,
) -> None:
    create_response = client.post("/api/users", json={"name": "Usuario Web"})
    user_id = create_response.json()["id"]

    response = client.post(
        "/api/me/telegram-link-code",
        headers={"X-User-Id": user_id},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["code"]
    assert body["expires_at"]


def test_create_transaction_endpoint_accepts_x_telegram_id(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/transactions",
        headers={"X-Telegram-Id": "223344"},
        json={
            "type": "expense",
            "amount": "19.90",
            "category": "transporte",
            "description": "Metro",
            "date": "2026-03-11",
        },
    )

    assert response.status_code == 201
    assert response.json()["category"] == "transporte"
    assert response.json()["type"] == "expense"


def test_transactions_are_isolated_between_users(
    client: TestClient,
    session_factory: sessionmaker[Session],
    user: User,
) -> None:
    with session_factory() as session:
        other_user = User(name="Outro", telegram_id=777888)
        session.add(other_user)
        session.flush()
        session.add_all(
            [
                Transaction(
                    user_id=user.id,
                    type=TransactionType.EXPENSE,
                    amount=Decimal("25.00"),
                    category="alimentacao",
                    description="Cafe",
                    date=date(2026, 3, 11),
                ),
                Transaction(
                    user_id=other_user.id,
                    type=TransactionType.EXPENSE,
                    amount=Decimal("99.00"),
                    category="lazer",
                    description="Cinema",
                    date=date(2026, 3, 11),
                ),
            ]
        )
        session.commit()
        session.refresh(other_user)

    first_response = client.get(
        "/api/transactions?limit=10",
        headers={"X-User-Id": str(user.id)},
    )
    second_response = client.get(
        "/api/transactions?limit=10",
        headers={"X-User-Id": str(other_user.id)},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert len(first_response.json()["items"]) == 1
    assert first_response.json()["items"][0]["description"] == "Cafe"
    assert len(second_response.json()["items"]) == 1
    assert second_response.json()["items"][0]["description"] == "Cinema"


def test_month_summary_is_isolated_between_users(
    client: TestClient,
    session_factory: sessionmaker[Session],
    user: User,
) -> None:
    with session_factory() as session:
        other_user = User(name="Resumo Outro", telegram_id=888999)
        session.add(other_user)
        session.flush()
        session.add_all(
            [
                Transaction(
                    user_id=user.id,
                    type=TransactionType.EXPENSE,
                    amount=Decimal("50.00"),
                    category="lazer",
                    description="Cinema",
                    date=date(2026, 3, 10),
                ),
                Transaction(
                    user_id=other_user.id,
                    type=TransactionType.EXPENSE,
                    amount=Decimal("300.00"),
                    category="moradia",
                    description="Aluguel",
                    date=date(2026, 3, 10),
                ),
            ]
        )
        session.commit()
        session.refresh(other_user)

    first_response = client.get(
        "/api/summary/month?month=3&year=2026",
        headers={"X-User-Id": str(user.id)},
    )
    second_response = client.get(
        "/api/summary/month?month=3&year=2026",
        headers={"X-User-Id": str(other_user.id)},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["total_expenses"] == "50.00"
    assert second_response.json()["total_expenses"] == "300.00"


def test_list_transactions_endpoint(
    client: TestClient,
    session_factory: sessionmaker[Session],
    user: User,
) -> None:
    with session_factory() as session:
        session.add(
            Transaction(
                user_id=user.id,
                type=TransactionType.EXPENSE,
                amount=Decimal("25.00"),
                category="alimentacao",
                description="Cafe",
                date=date(2026, 3, 11),
            )
        )
        session.commit()

    response = client.get(
        "/api/transactions?limit=10",
        headers={"X-User-Id": str(user.id)},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 1
    assert body["items"][0]["category"] == "alimentacao"


def test_get_month_summary_endpoint(
    client: TestClient,
    session_factory: sessionmaker[Session],
    user: User,
) -> None:
    with session_factory() as session:
        session.add(
            Transaction(
                user_id=user.id,
                type=TransactionType.EXPENSE,
                amount=Decimal("50.00"),
                category="lazer",
                description="Cinema",
                date=date(2026, 3, 10),
            )
        )
        session.commit()

    response = client.get(
        "/api/summary/month?month=3&year=2026",
        headers={"X-User-Id": str(user.id)},
    )

    assert response.status_code == 200
    assert response.json()["total_expenses"] == "50.00"
    assert response.json()["insights"]
    assert "lazer" in " ".join(response.json()["insights"]).lower()


def test_import_transactions_csv_endpoint(
    client: TestClient,
    user: User,
) -> None:
    csv_content = "\n".join(
        [
            "type,amount,category,description,date",
            "expense,15.50,transporte,Onibus,2026-03-01",
            "income,300.00,freelance,Projeto,2026-03-02",
            "expense,abc,alimentacao,Almoco,2026-03-03",
        ]
    )

    response = client.post(
        "/api/transactions/import",
        headers={"X-User-Id": str(user.id)},
        files={"file": ("import.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["imported_count"] == 2
    assert body["skipped_count"] == 1
    assert len(body["transactions"]) == 2
    assert body["errors"][0]["line_number"] == 4
