from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import StrEnum

from sqlalchemy.orm import Session

from app.agents import CategorizationAgent, IngestionAgent, RouterAgent
from app.agents.types import (
    CategorizationAgentInput,
    IngestionAgentInput,
    IntentType,
    RouterAgentInput,
)
from app.models.enums import TransactionType
from app.schemas.transaction import TransactionCreateRequest, TransactionResponse
from app.services.agent_log_service import log_agent_execution
from app.services.transaction_service import save_transaction_service


class ExpenseMessageFlowStatus(StrEnum):
    COMPLETED = "completed"
    CONFIRMATION_REQUIRED = "confirmation_required"
    CLARIFICATION_REQUIRED = "clarification_required"
    NOT_HANDLED = "not_handled"
    ERROR = "error"


@dataclass(slots=True)
class PendingExpenseDraft:
    user_id: uuid.UUID
    amount: Decimal
    category: str
    date: date
    description: str | None
    original_message: str


@dataclass(slots=True)
class ExpenseMessageFlowResult:
    status: ExpenseMessageFlowStatus
    message: str
    transaction: TransactionResponse | None = None


@dataclass(slots=True)
class ExpenseMessageFlowService:
    router_agent: RouterAgent = field(default_factory=RouterAgent)
    ingestion_agent: IngestionAgent = field(default_factory=IngestionAgent)
    categorization_agent: CategorizationAgent = field(default_factory=CategorizationAgent)
    pending_confirmations: dict[uuid.UUID, PendingExpenseDraft] = field(default_factory=dict)

    def process_message(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        message: str,
        reference_date: date | None = None,
    ) -> ExpenseMessageFlowResult:
        cleaned_message = message.strip()
        if not cleaned_message:
            return ExpenseMessageFlowResult(
                status=ExpenseMessageFlowStatus.CLARIFICATION_REQUIRED,
                message="Envie uma mensagem com a despesa que deseja registrar.",
            )

        try:
            confirmation_result = self._handle_pending_confirmation(
                session=session,
                user_id=user_id,
                message=cleaned_message,
            )
            if confirmation_result is not None:
                return confirmation_result

            router_input = RouterAgentInput(message=cleaned_message)
            routed = self.router_agent.run(router_input)
            log_agent_execution(
                session,
                agent_name=self.router_agent.name,
                input_payload=router_input,
                output_payload=routed,
            )
            if routed.intent != IntentType.REGISTER_EXPENSE:
                return ExpenseMessageFlowResult(
                    status=ExpenseMessageFlowStatus.NOT_HANDLED,
                    message="No momento eu so registro despesas por mensagem.",
                )

            ingestion_input = IngestionAgentInput(
                message=cleaned_message,
                reference_date=reference_date or date.today(),
            )
            ingestion = self.ingestion_agent.run(ingestion_input)
            log_agent_execution(
                session,
                agent_name=self.ingestion_agent.name,
                input_payload=ingestion_input,
                output_payload=ingestion,
            )

            if ingestion.type not in (None, TransactionType.EXPENSE):
                return ExpenseMessageFlowResult(
                    status=ExpenseMessageFlowStatus.NOT_HANDLED,
                    message="Essa mensagem nao parece ser uma despesa.",
                )

            if ingestion.amount is None:
                return ExpenseMessageFlowResult(
                    status=ExpenseMessageFlowStatus.CLARIFICATION_REQUIRED,
                    message="Nao consegui identificar o valor da despesa. Informe o valor para continuar.",
                )

            if ingestion.date is None:
                return ExpenseMessageFlowResult(
                    status=ExpenseMessageFlowStatus.CLARIFICATION_REQUIRED,
                    message="Nao consegui identificar a data da despesa. Informe a data para continuar.",
                )

            categorization_input = CategorizationAgentInput(
                type=TransactionType.EXPENSE,
                description=ingestion.description,
                suggested_category=ingestion.category,
            )
            categorized = self.categorization_agent.run(categorization_input)
            log_agent_execution(
                session,
                agent_name=self.categorization_agent.name,
                input_payload=categorization_input,
                output_payload=categorized,
            )

            if self._requires_confirmation(ingestion.category, categorized.category):
                self.pending_confirmations[user_id] = PendingExpenseDraft(
                    user_id=user_id,
                    amount=ingestion.amount,
                    category=categorized.category,
                    date=ingestion.date,
                    description=ingestion.description,
                    original_message=cleaned_message,
                )
                return ExpenseMessageFlowResult(
                    status=ExpenseMessageFlowStatus.CONFIRMATION_REQUIRED,
                    message=(
                        f"Voce deseja registrar uma despesa de R${ingestion.amount} "
                        f"em {categorized.category} na data {ingestion.date}? Responda 'sim' para confirmar."
                    ),
                )

            transaction = save_transaction_service(
                session,
                user_id=user_id,
                payload=TransactionCreateRequest(
                    type=TransactionType.EXPENSE,
                    amount=ingestion.amount,
                    category=categorized.category,
                    description=ingestion.description,
                    date=ingestion.date,
                ),
            )
            return ExpenseMessageFlowResult(
                status=ExpenseMessageFlowStatus.COMPLETED,
                message=f"Despesa registrada: R${transaction.amount} em {transaction.category}.",
                transaction=transaction,
            )
        except Exception as exc:
            log_agent_execution(
                session,
                agent_name="ExpenseMessageFlow",
                input_payload={"user_id": str(user_id), "message": cleaned_message},
                output_payload={"error": str(exc)},
            )
            return ExpenseMessageFlowResult(
                status=ExpenseMessageFlowStatus.ERROR,
                message="Ocorreu um erro ao registrar a transacao. Tente novamente.",
            )

    def _handle_pending_confirmation(
        self,
        session: Session,
        *,
        user_id: uuid.UUID,
        message: str,
    ) -> ExpenseMessageFlowResult | None:
        draft = self.pending_confirmations.get(user_id)
        if draft is None:
            return None

        normalized = message.strip().casefold()
        if normalized in {"sim", "confirmo", "confirmar", "ok"}:
            transaction = save_transaction_service(
                session,
                user_id=user_id,
                payload=TransactionCreateRequest(
                    type=TransactionType.EXPENSE,
                    amount=draft.amount,
                    category=draft.category,
                    description=draft.description,
                    date=draft.date,
                ),
            )
            del self.pending_confirmations[user_id]
            return ExpenseMessageFlowResult(
                status=ExpenseMessageFlowStatus.COMPLETED,
                message=f"Despesa registrada: R${transaction.amount} em {transaction.category}.",
                transaction=transaction,
            )

        if normalized in {"nao", "cancelar", "cancela"}:
            del self.pending_confirmations[user_id]
            return ExpenseMessageFlowResult(
                status=ExpenseMessageFlowStatus.CLARIFICATION_REQUIRED,
                message="Registro cancelado. Envie a despesa novamente com mais detalhes.",
            )

        return ExpenseMessageFlowResult(
            status=ExpenseMessageFlowStatus.CONFIRMATION_REQUIRED,
            message="Responda 'sim' para confirmar ou 'cancelar' para descartar essa despesa.",
        )

    def _requires_confirmation(
        self,
        extracted_category: str | None,
        validated_category: str,
    ) -> bool:
        if extracted_category is None:
            return True
        return extracted_category != validated_category
