from __future__ import annotations

from datetime import date
from decimal import Decimal

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.models import AgentLog, Transaction, User
from app.models.enums import TransactionType


def test_expense_flow_endpoint_registers_expense(
    client: TestClient,
    session_factory: sessionmaker[Session],
    user: User,
) -> None:
    response = client.post(
        "/api/messages/transaction",
        headers={"X-User-Id": str(user.id)},
        json={"message": "Gastei 35 no Uber ontem"},
    )

    with session_factory() as session:
        transaction = session.scalar(select(Transaction).where(Transaction.user_id == user.id))
        logs = session.scalars(select(AgentLog)).all()

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "registrar_despesa"
    assert body["parsed_data"]["category"] == "transporte"
    assert body["parsed_data"]["description"] == "Uber"
    assert body["saved_transaction"] is not None
    assert body["saved_transaction"]["amount"] == "35.00"
    assert "Despesa registrada" in body["response_message"]
    assert transaction is not None
    assert transaction.category == "transporte"
    assert len(logs) == 3


def test_expense_flow_endpoint_maps_food_keywords_and_cleans_description(
    client: TestClient,
    session_factory: sessionmaker[Session],
    user: User,
) -> None:
    response = client.post(
        "/api/messages/transaction",
        headers={"X-User-Id": str(user.id)},
        json={"message": "Gastei 42 no almoço hoje"},
    )

    with session_factory() as session:
        transaction = session.scalar(select(Transaction).where(Transaction.user_id == user.id))

    assert response.status_code == 200
    body = response.json()
    assert body["parsed_data"]["category"] == "alimentação"
    assert body["parsed_data"]["description"] == "almoço"
    assert body["saved_transaction"] is not None
    assert body["saved_transaction"]["category"] == "alimentação"
    assert transaction is not None
    assert transaction.category == "alimentação"


def test_expense_flow_endpoint_maps_cafe_to_alimentacao(
    client: TestClient,
    user: User,
) -> None:
    response = client.post(
        "/api/messages/transaction",
        headers={"X-User-Id": str(user.id)},
        json={"message": "Paguei 9 no café hoje"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["parsed_data"]["category"] == "alimentação"
    assert body["parsed_data"]["description"] == "café"
    assert body["saved_transaction"] is not None
    assert body["saved_transaction"]["category"] == "alimentação"


def test_expense_flow_endpoint_maps_metro_to_transporte(
    client: TestClient,
    user: User,
) -> None:
    response = client.post(
        "/api/messages/transaction",
        headers={"X-User-Id": str(user.id)},
        json={"message": "Paguei 12 no metrô hoje"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["parsed_data"]["category"] == "transporte"
    assert body["parsed_data"]["description"] == "metrô"
    assert body["saved_transaction"] is not None
    assert body["saved_transaction"]["category"] == "transporte"


def test_expense_flow_endpoint_returns_clarification_for_ambiguous_category(
    client: TestClient,
    session_factory: sessionmaker[Session],
    user: User,
) -> None:
    response = client.post(
        "/api/messages/transaction",
        headers={"X-User-Id": str(user.id)},
        json={"message": "Gastei 35 ontem"},
    )

    with session_factory() as session:
        transaction = session.scalar(select(Transaction).where(Transaction.user_id == user.id))

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "registrar_despesa"
    assert body["saved_transaction"] is None
    assert "ambigua" in body["response_message"]
    assert transaction is None


def test_income_flow_endpoint_registers_income_message(
    client: TestClient,
    session_factory: sessionmaker[Session],
    user: User,
) -> None:
    response = client.post(
        "/api/messages/transaction",
        headers={"X-User-Id": str(user.id)},
        json={"message": "Recebi 2500 de salário hoje"},
    )

    with session_factory() as session:
        transaction = session.scalar(select(Transaction).where(Transaction.user_id == user.id))

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "registrar_receita"
    assert body["parsed_data"]["type"] == "income"
    assert body["parsed_data"]["category"] == "salário"
    assert body["parsed_data"]["description"] == "salário"
    assert body["saved_transaction"] is not None
    assert body["saved_transaction"]["amount"] == "2500.00"
    assert body["saved_transaction"]["type"] == "income"
    assert body["saved_transaction"]["category"] == "salário"
    assert "Receita registrada" in body["response_message"]
    assert transaction is not None
    assert transaction.type.value == "income"


def test_income_flow_endpoint_returns_clarification_for_ambiguous_income_category(
    client: TestClient,
    session_factory: sessionmaker[Session],
    user: User,
) -> None:
    response = client.post(
        "/api/messages/transaction",
        headers={"X-User-Id": str(user.id)},
        json={"message": "Recebi 2000 hoje"},
    )

    with session_factory() as session:
        transaction = session.scalar(select(Transaction).where(Transaction.user_id == user.id))

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "registrar_receita"
    assert body["saved_transaction"] is None
    assert "ambigua" in body["response_message"]
    assert transaction is None


def test_month_summary_endpoint_flow_returns_current_month_summary(
    client: TestClient,
    session_factory: sessionmaker[Session],
    user: User,
) -> None:
    with session_factory() as session:
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

    response = client.post(
        "/api/messages/transaction",
        headers={"X-User-Id": str(user.id)},
        json={"message": "Quanto gastei esse mês?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "consultar_resumo"
    assert body["saved_transaction"] is None
    assert body["parsed_data"]["total_expenses"] == "120.00"
    assert body["parsed_data"]["total_income"] == "500.00"
    assert body["parsed_data"]["balance"] == "380.00"
    assert body["parsed_data"]["insights"]
    assert "Resumo do mes" in body["response_message"]
    assert "alimentação" in body["response_message"].lower()


def test_recent_history_flow_returns_recent_transactions(
    client: TestClient,
    session_factory: sessionmaker[Session],
    user: User,
) -> None:
    with session_factory() as session:
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

    response = client.post(
        "/api/messages/transaction",
        headers={"X-User-Id": str(user.id)},
        json={"message": "Mostre minhas últimas transações"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "consultar_historico"
    assert body["saved_transaction"] is None
    assert len(body["parsed_data"]["items"]) == 2
    assert "Suas transacoes mais recentes" in body["response_message"]


def test_recent_history_flow_filters_expenses_when_user_asks_for_gastos(
    client: TestClient,
    session_factory: sessionmaker[Session],
    user: User,
) -> None:
    with session_factory() as session:
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

    response = client.post(
        "/api/messages/transaction",
        headers={"X-User-Id": str(user.id)},
        json={"message": "Últimas 5 despesas"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "consultar_historico"
    assert len(body["parsed_data"]["items"]) == 1
    assert body["parsed_data"]["items"][0]["type"] == "expense"


def test_insights_flow_returns_current_month_insights(
    client: TestClient,
    session_factory: sessionmaker[Session],
    user: User,
) -> None:
    with session_factory() as session:
        session.add_all(
            [
                Transaction(
                    user_id=user.id,
                    type=TransactionType.EXPENSE,
                    amount=Decimal("90.00"),
                    category="transporte",
                    description="uber",
                    date=date.today(),
                ),
                Transaction(
                    user_id=user.id,
                    type=TransactionType.EXPENSE,
                    amount=Decimal("150.00"),
                    category="alimentação",
                    description="jantar",
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

    response = client.post(
        "/api/messages/transaction",
        headers={"X-User-Id": str(user.id)},
        json={"message": "Qual foi meu maior gasto do mês?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "consultar_insights"
    assert body["saved_transaction"] is None
    assert body["parsed_data"]["insights"]
    assert "Insights financeiros" in body["response_message"]
    assert "alimentação" in body["response_message"].lower()


def test_recent_history_flow_is_isolated_between_users(
    client: TestClient,
    session_factory: sessionmaker[Session],
    user: User,
) -> None:
    with session_factory() as session:
        other_user = User(name="Outro historico", telegram_id=222333)
        session.add(other_user)
        session.flush()
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
                    user_id=other_user.id,
                    type=TransactionType.EXPENSE,
                    amount=Decimal("400.00"),
                    category="moradia",
                    description="aluguel",
                    date=date.today(),
                ),
            ]
        )
        session.commit()
        session.refresh(other_user)

    response = client.post(
        "/api/messages/transaction",
        headers={"X-User-Id": str(user.id)},
        json={"message": "Mostre minhas últimas transações"},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["parsed_data"]["items"]) == 1
    assert body["parsed_data"]["items"][0]["description"] == "almoço"


def test_month_summary_and_insights_are_isolated_between_users(
    client: TestClient,
    session_factory: sessionmaker[Session],
    user: User,
) -> None:
    with session_factory() as session:
        other_user = User(name="Outro insight", telegram_id=444555)
        session.add(other_user)
        session.flush()
        session.add_all(
            [
                Transaction(
                    user_id=user.id,
                    type=TransactionType.EXPENSE,
                    amount=Decimal("70.00"),
                    category="transporte",
                    description="uber",
                    date=date.today(),
                ),
                Transaction(
                    user_id=other_user.id,
                    type=TransactionType.EXPENSE,
                    amount=Decimal("500.00"),
                    category="moradia",
                    description="aluguel",
                    date=date.today(),
                ),
            ]
        )
        session.commit()
        session.refresh(other_user)

    response = client.post(
        "/api/messages/transaction",
        headers={"X-User-Id": str(user.id)},
        json={"message": "Analise minhas finanças deste mês"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "consultar_insights"
    assert "transporte" in body["response_message"].lower()
    assert "moradia" not in body["response_message"].lower()
