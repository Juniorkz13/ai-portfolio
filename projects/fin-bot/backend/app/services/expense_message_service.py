from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy.orm import Session

from app.agents.categorization_agent import CategorizationAgent
from app.agents.ingestion_agent import IngestionAgent
from app.agents.router_agent import RouterAgent
from app.agents.types import CategorizationAgentInput, IngestionAgentInput, IntentType, RouterAgentInput
from app.models.enums import TransactionType
from app.schemas.expense_flow import ExpenseMessageResponse
from app.schemas.transaction import TransactionCreateRequest
from app.services.agent_log_service import log_agent_execution
from app.services.transaction_service import (
    get_month_summary_service,
    list_recent_transactions_service,
    save_transaction_service,
)


@dataclass(slots=True)
class ExpenseMessageService:
    router_agent: RouterAgent = field(default_factory=RouterAgent)
    ingestion_agent: IngestionAgent = field(default_factory=IngestionAgent)
    categorization_agent: CategorizationAgent = field(default_factory=CategorizationAgent)

    def process(self, session: Session, *, user_id: uuid.UUID, message: str) -> ExpenseMessageResponse:
        router_input = RouterAgentInput(message=message)
        routed = self.router_agent.run(router_input)
        log_agent_execution(
            session,
            agent_name=self.router_agent.name,
            input_payload=router_input,
            output_payload=routed,
        )

        if routed.intent == IntentType.GET_SUMMARY:
            return self._process_month_summary(session, user_id=user_id, intent=routed.intent)
        if routed.intent == IntentType.GET_INSIGHTS:
            return self._process_month_insights(session, user_id=user_id, intent=routed.intent)
        if routed.intent == IntentType.GET_HISTORY:
            return self._process_recent_history(
                session,
                user_id=user_id,
                intent=routed.intent,
                message=message,
            )

        if routed.intent not in (IntentType.REGISTER_EXPENSE, IntentType.REGISTER_INCOME):
            return ExpenseMessageResponse(
                intent=routed.intent.value,
                parsed_data={},
                saved_transaction=None,
                response_message="Esta rota aceita apenas mensagens de despesa, receita, resumo mensal, insights financeiros ou historico recente.",
            )

        ingestion_input = IngestionAgentInput(message=message, reference_date=date.today())
        parsed = self.ingestion_agent.run(ingestion_input)
        log_agent_execution(
            session,
            agent_name=self.ingestion_agent.name,
            input_payload=ingestion_input,
            output_payload=parsed,
        )

        transaction_type = self._resolve_transaction_type(routed.intent, parsed.type)
        if transaction_type is None:
            return ExpenseMessageResponse(
                intent=routed.intent.value,
                parsed_data={},
                saved_transaction=None,
                response_message="Nao consegui identificar se a mensagem representa uma despesa ou uma receita.",
            )

        parsed_data: dict[str, object | None] = {
            "type": transaction_type.value,
            "amount": str(parsed.amount) if parsed.amount is not None else None,
            "category": parsed.category,
            "description": parsed.description,
            "date": parsed.date.isoformat() if parsed.date is not None else None,
        }

        if parsed.amount is None or parsed.date is None:
            return ExpenseMessageResponse(
                intent=routed.intent.value,
                parsed_data=parsed_data,
                saved_transaction=None,
                response_message="Nao consegui identificar todos os dados da transacao. Envie valor e data claramente.",
            )

        categorization_input = CategorizationAgentInput(
            type=transaction_type,
            description=parsed.description,
            suggested_category=parsed.category,
        )
        categorized = self.categorization_agent.run(categorization_input)
        log_agent_execution(
            session,
            agent_name=self.categorization_agent.name,
            input_payload=categorization_input,
            output_payload=categorized,
        )
        parsed_data["category"] = categorized.category

        if parsed.category is None and categorized.category == "outros":
            return ExpenseMessageResponse(
                intent=routed.intent.value,
                parsed_data=parsed_data,
                saved_transaction=None,
                response_message="A categoria ficou ambigua. Envie mais detalhes da transacao antes de registrar.",
            )

        transaction = save_transaction_service(
            session,
            user_id=user_id,
            payload=TransactionCreateRequest(
                type=transaction_type,
                amount=parsed.amount,
                category=categorized.category,
                description=parsed.description,
                date=parsed.date,
            ),
        )

        response_prefix = "Despesa" if transaction_type == TransactionType.EXPENSE else "Receita"

        return ExpenseMessageResponse(
            intent=routed.intent.value,
            parsed_data=parsed_data,
            saved_transaction=transaction,
            response_message=f"{response_prefix} registrada: R${transaction.amount} em {transaction.category}.",
        )

    def _resolve_transaction_type(
        self,
        intent: IntentType,
        parsed_type: TransactionType | None,
    ) -> TransactionType | None:
        expected_type = {
            IntentType.REGISTER_EXPENSE: TransactionType.EXPENSE,
            IntentType.REGISTER_INCOME: TransactionType.INCOME,
        }.get(intent)

        if expected_type is None:
            return None
        if parsed_type is None:
            return expected_type
        if parsed_type != expected_type:
            return None
        return parsed_type

    def _process_month_summary(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        intent: IntentType,
    ) -> ExpenseMessageResponse:
        today = date.today()
        summary = get_month_summary_service(
            session,
            user_id=user_id,
            month=today.month,
            year=today.year,
        )
        parsed_data = {
            "month": today.month,
            "year": today.year,
            "total_expenses": str(summary.total_expenses),
            "total_income": str(summary.total_income),
            "balance": str(summary.balance),
            "insights": summary.insights,
        }
        response_lines = [
            f"Resumo do mes: receitas R${summary.total_income}, "
            f"despesas R${summary.total_expenses} e saldo R${summary.balance}."
        ]
        if summary.insights:
            response_lines.append("Insights:")
            response_lines.extend(f"- {insight}" for insight in summary.insights)
        return ExpenseMessageResponse(
            intent=intent.value,
            parsed_data=parsed_data,
            saved_transaction=None,
            response_message="\n".join(response_lines),
        )

    def _process_month_insights(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        intent: IntentType,
    ) -> ExpenseMessageResponse:
        today = date.today()
        summary = get_month_summary_service(
            session,
            user_id=user_id,
            month=today.month,
            year=today.year,
        )
        parsed_data = {
            "month": today.month,
            "year": today.year,
            "total_expenses": str(summary.total_expenses),
            "total_income": str(summary.total_income),
            "balance": str(summary.balance),
            "insights": summary.insights,
        }
        if not summary.insights:
            response_message = (
                f"Ainda nao encontrei insights relevantes para {today.month:02d}/{today.year}. "
                "Registre mais transacoes neste mes para eu analisar melhor."
            )
        else:
            response_lines = [
                f"Insights financeiros de {today.month:02d}/{today.year}:",
            ]
            response_lines.extend(f"- {insight}" for insight in summary.insights)
            response_message = "\n".join(response_lines)

        return ExpenseMessageResponse(
            intent=intent.value,
            parsed_data=parsed_data,
            saved_transaction=None,
            response_message=response_message,
        )

    def _process_recent_history(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        intent: IntentType,
        message: str,
    ) -> ExpenseMessageResponse:
        limit = self._extract_history_limit(message)
        transactions = list_recent_transactions_service(
            session,
            user_id=user_id,
            limit=limit,
        )

        normalized_message = message.casefold()
        if any(keyword in normalized_message for keyword in ("gastos", "despesas", "despesa")):
            transactions = [
                transaction for transaction in transactions if transaction.type == TransactionType.EXPENSE
            ]

        items = [
            {
                "date": transaction.date.isoformat(),
                "type": transaction.type.value,
                "amount": str(transaction.amount),
                "category": transaction.category,
                "description": transaction.description,
            }
            for transaction in transactions[:limit]
        ]

        if not items:
            return ExpenseMessageResponse(
                intent=intent.value,
                parsed_data={"limit": limit, "items": []},
                saved_transaction=None,
                response_message="Nao encontrei transacoes recentes para este usuario.",
            )

        response_lines = ["Suas transacoes mais recentes:"]
        for item in items:
            response_lines.append(
                f"- {item['date']}: {item['type']} de R${item['amount']} em {item['category']}"
            )

        return ExpenseMessageResponse(
            intent=intent.value,
            parsed_data={"limit": limit, "items": items},
            saved_transaction=None,
            response_message="\n".join(response_lines),
        )

    def _extract_history_limit(self, message: str) -> int:
        match = re.search(r"\b(\d{1,2})\b", message)
        if not match:
            return 5
        limit = int(match.group(1))
        return max(1, min(limit, 10))
