from __future__ import annotations

from decimal import Decimal

from app.agents.types import AnalyticsAgentOutput, RecommendationAgentOutput
from app.schemas.summary import MonthSummaryResponse
from app.schemas.transaction import CsvImportResponse, TransactionResponse


def build_transaction_confirmation(transaction: TransactionResponse) -> str:
    return (
        f"Transacao registrada: {transaction.type.value} de {transaction.amount} "
        f"em {transaction.category} na data {transaction.date}."
    )


def build_month_summary_message(
    summary: MonthSummaryResponse,
    analytics: AnalyticsAgentOutput,
    recommendations: RecommendationAgentOutput,
) -> str:
    lines = [
        f"Receitas: {summary.total_income}",
        f"Despesas: {summary.total_expenses}",
        f"Saldo: {summary.balance}",
        analytics.summary_text,
    ]

    if summary.expenses_by_category:
        top_categories = ", ".join(
            f"{item.category}: {item.total}" for item in summary.expenses_by_category[:3]
        )
        lines.append(f"Categorias com mais gasto: {top_categories}.")

    if recommendations.recommendations:
        lines.append(f"Recomendacao: {recommendations.recommendations[0]}")

    return "\n".join(lines)


def build_recent_transactions_message(transactions: list[TransactionResponse]) -> str:
    if not transactions:
        return "Nao encontrei transacoes recentes para este usuario."

    lines = ["Ultimas transacoes:"]
    for item in transactions:
        lines.append(
            f"- {item.date}: {item.type.value} de {item.amount} em {item.category}"
        )
    return "\n".join(lines)


def build_import_result_message(result: CsvImportResponse) -> str:
    return f"Importacao concluida com {result.imported_count} transacoes processadas."


def build_unknown_intent_message() -> str:
    return (
        "Nao entendi sua solicitacao. Tente algo como "
        "'Gastei 45 no Uber' ou 'Quanto gastei esse mes?'."
    )


def normalize_decimal_text(value: Decimal | None) -> str:
    return str(value) if value is not None else "0"
