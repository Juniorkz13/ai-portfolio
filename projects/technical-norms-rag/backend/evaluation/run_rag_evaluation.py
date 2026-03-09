from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


@dataclass
class EvalCase:
    """Represents one evaluation question and its expected terms."""

    case_id: str
    question: str
    expected_terms: list[str]
    filters: dict[str, Any]


def load_cases(path: Path) -> list[EvalCase]:
    """Load evaluation cases from JSON file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for item in raw:
        cases.append(
            EvalCase(
                case_id=str(item["id"]),
                question=str(item["question"]),
                expected_terms=[str(term) for term in item.get("expected_terms", [])],
                filters=dict(item.get("filters", {})),
            )
        )
    return cases


def call_chat_api(
    *,
    api_base_url: str,
    question: str,
    top_k: int,
    filters: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    """Call `/chat` endpoint and return parsed JSON payload."""
    payload: dict[str, Any] = {"question": question, "top_k": top_k}
    payload.update(filters)

    response = requests.post(
        f"{api_base_url.rstrip('/')}/chat",
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def call_chat_service(
    *,
    question: str,
    top_k: int,
    filters: dict[str, Any],
) -> dict[str, Any]:
    """Call chat service directly (without HTTP)."""
    from app.core.database import get_db
    from app.services.chat_service import ChatService
    from app.services.llm.gemini_client import GeminiClient
    from app.services.retrieval_service import RetrievalService

    db_gen = get_db()
    db = next(db_gen)
    try:
        retrieval = RetrievalService(db=db)
        llm = GeminiClient()
        chat = ChatService(retrieval_service=retrieval, llm_client=llm)
        return chat.answer(question=question, top_k=top_k, filters=filters or None)
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


def evaluate_case(case: EvalCase, response: dict[str, Any]) -> dict[str, Any]:
    """Compute a simple pass/review status based on expected terms."""
    answer = str(response.get("answer", ""))
    explanation = str(response.get("explanation", ""))
    combined_text = f"{answer} {explanation}".lower()

    expected = [term.lower() for term in case.expected_terms]
    matched = [term for term in expected if term in combined_text]

    status = "pass" if (not expected or len(matched) == len(expected)) else "review"

    return {
        "case_id": case.case_id,
        "question": case.question,
        "expected_terms": case.expected_terms,
        "matched_terms": matched,
        "status": status,
        "manual_observation": "",
        "generated_answer": answer,
        "generated_explanation": explanation,
        "sources": response.get("sources", []),
    }


def run_evaluation(
    *,
    mode: str,
    cases_path: Path,
    output_path: Path,
    api_base_url: str,
    top_k: int,
    timeout_seconds: int,
) -> None:
    """Run local RAG evaluation and save a JSON report."""
    cases = load_cases(cases_path)
    results: list[dict[str, Any]] = []

    for case in cases:
        try:
            if mode == "api":
                response = call_chat_api(
                    api_base_url=api_base_url,
                    question=case.question,
                    top_k=top_k,
                    filters=case.filters,
                    timeout_seconds=timeout_seconds,
                )
            else:
                response = call_chat_service(
                    question=case.question,
                    top_k=top_k,
                    filters=case.filters,
                )
            results.append(evaluate_case(case, response))
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "expected_terms": case.expected_terms,
                    "matched_terms": [],
                    "status": "error",
                    "manual_observation": str(exc),
                    "generated_answer": "",
                    "generated_explanation": "",
                    "sources": [],
                }
            )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "total_cases": len(results),
        "pass_cases": len([r for r in results if r["status"] == "pass"]),
        "review_cases": len([r for r in results if r["status"] == "review"]),
        "error_cases": len([r for r in results if r["status"] == "error"]),
    }

    report = {"summary": summary, "results": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Evaluation completed. Report saved at: {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run practical RAG evaluation cases.")
    parser.add_argument("--mode", choices=["api", "service"], default="api")
    parser.add_argument(
        "--cases",
        default="evaluation/test_cases.json",
        help="Path to JSON file containing evaluation cases.",
    )
    parser.add_argument(
        "--output",
        default=(
            f"evaluation/results/eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        ),
        help="Path to output JSON report.",
    )
    parser.add_argument("--api-base-url", default="http://localhost:8000")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        mode=args.mode,
        cases_path=Path(args.cases),
        output_path=Path(args.output),
        api_base_url=args.api_base_url,
        top_k=args.top_k,
        timeout_seconds=args.timeout,
    )
