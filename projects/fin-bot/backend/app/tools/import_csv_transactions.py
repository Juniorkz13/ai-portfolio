from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.tools._compat import tool
from app.tools.save_transaction import save_transaction
from app.tools.types import (
    CsvImportInput,
    ImportCsvErrorRow,
    ImportCsvTransactionsOutput,
    ImportedTransactionRow,
    SaveTransactionInput,
    SessionFactory,
    load_csv_rows,
)


def import_csv_transactions(
    session: Session, payload: CsvImportInput
) -> ImportCsvTransactionsOutput:
    imported_rows: list[ImportedTransactionRow] = []
    error_rows: list[ImportCsvErrorRow] = []

    for line_number, row in enumerate(load_csv_rows(payload), start=2):
        try:
            save_payload = SaveTransactionInput(
                user_id=payload.user_id,
                type=(row.get("type") or "").strip(),
                amount=Decimal((row.get("amount") or "").strip()),
                category=(row.get("category") or "").strip(),
                description=(row.get("description") or "").strip() or None,
                date=date.fromisoformat((row.get("date") or "").strip()),
            )
            transaction = save_transaction(session, save_payload, commit=False)
            imported_rows.append(
                ImportedTransactionRow(line_number=line_number, transaction=transaction)
            )
        except (ArithmeticError, KeyError, TypeError, ValidationError, ValueError) as exc:
            error_rows.append(
                ImportCsvErrorRow(
                    line_number=line_number,
                    error=f"Linha invalida: {exc}",
                )
            )

    session.commit()

    return ImportCsvTransactionsOutput(
        imported_count=len(imported_rows),
        skipped_count=len(error_rows),
        transactions=imported_rows,
        errors=error_rows,
    )


def build_import_csv_transactions_tool(session_factory: SessionFactory):
    @tool("import_csv_transactions", args_schema=CsvImportInput)
    def import_csv_transactions_tool(
        user_id: str,
        csv_content: str | None = None,
        file_path: str | None = None,
    ) -> dict[str, object]:
        payload = CsvImportInput(
            user_id=user_id,
            csv_content=csv_content,
            file_path=file_path,
        )
        with session_factory() as session:
            result = import_csv_transactions(session, payload)
        return result.model_dump(mode="json")

    return import_csv_transactions_tool
