from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from app.models import AgentLog


def log_agent_execution(
    session: Session,
    *,
    agent_name: str,
    input_payload: Any,
    output_payload: Any,
) -> None:
    log_entry = AgentLog(
        agent_name=agent_name,
        input=_serialize_payload(input_payload),
        output=_serialize_payload(output_payload),
    )
    session.add(log_entry)
    session.commit()


def _serialize_payload(payload: Any) -> str:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json", by_alias=True)
    return json.dumps(payload, ensure_ascii=True, default=str)
