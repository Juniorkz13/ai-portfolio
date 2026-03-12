from __future__ import annotations

from datetime import date
from decimal import Decimal

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base, TransactionType, User
from app.tools.get_category_summary import get_category_summary
from app.tools.get_month_summary import get_month_summary
from app.tools.import_csv_transactions import import_csv_transactions
from app.tools.list_recent_transactions import list_recent_transactions
from app.tools.save_transaction import save_transaction
from app.tools.types import (
    CsvImportInput,
    GetCategorySummaryInput,
    ListRecentTransactionsInput,
    MonthSummaryInput,
    SaveTransactionInput,
)


def build_session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    return factory()


def create_user(session: Session) -> User:
    user = User(telegram_id=123456789, name="Jose")
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def test_save_and_list_recent_transactions() -> None:
    with build_session() as session:
        user = create_user(session)
        save_transaction(
            session,
            SaveTransactionInput(
                user_id=user.id,
                type=TransactionType.EXPENSE,
                amount=Decimal("25.90"),
                category="alimentacao",
                description="Almoco",
                date=date(2026, 3, 10),
            ),
        )

        results = list_recent_transactions(
            session,
            ListRecentTransactionsInput(user_id=user.id, limit=5),
        )

        assert len(results) == 1
        assert results[0].category == "alimentacao"


def test_month_and_category_summary() -> None:
    with build_session() as session:
        user = create_user(session)
        save_transaction(
            session,
            SaveTransactionInput(
                user_id=user.id,
                type=TransactionType.EXPENSE,
                amount=Decimal("120.00"),
                category="moradia",
                description=None,
                date=date(2026, 3, 2),
            ),
        )
        save_transaction(
            session,
            SaveTransactionInput(
                user_id=user.id,
                type=TransactionType.INCOME,
                amount=Decimal("500.00"),
                category="salario",
                description=None,
                date=date(2026, 3, 5),
            ),
        )

        month_summary = get_month_summary(
            session,
            MonthSummaryInput(user_id=user.id, month=3, year=2026),
        )
        category_summary = get_category_summary(
            session,
            GetCategorySummaryInput(user_id=user.id, month=3, year=2026),
        )

        assert month_summary.total_expenses == Decimal("120.00")
        assert month_summary.total_income == Decimal("500.00")
        assert month_summary.balance == Decimal("380.00")
        assert category_summary[0].category == "moradia"


def test_import_csv_transactions() -> None:
    csv_content = "\n".join(
        [
            "type,amount,category,description,date",
            "expense,10.50,transporte,Onibus,2026-03-01",
            "income,200.00,freelance,Projeto,2026-03-02",
        ]
    )

    with build_session() as session:
        user = create_user(session)
        result = import_csv_transactions(
            session,
            CsvImportInput(user_id=user.id, csv_content=csv_content),
        )

        assert result.imported_count == 2
        assert result.skipped_count == 0
        assert result.transactions[0].line_number == 2


def test_import_csv_transactions_skips_invalid_rows() -> None:
    csv_content = "\n".join(
        [
            "type,amount,category,description,date",
            "expense,10.50,transporte,Onibus,2026-03-01",
            "expense,abc,alimentacao,Almoco,2026-03-02",
        ]
    )

    with build_session() as session:
        user = create_user(session)
        result = import_csv_transactions(
            session,
            CsvImportInput(user_id=user.id, csv_content=csv_content),
        )

        assert result.imported_count == 1
        assert result.skipped_count == 1
        assert result.errors[0].line_number == 3
