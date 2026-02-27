from typing import TypedDict, Literal, List, Optional


class LegalDocument(TypedDict):
    source: str
    title: str
    content: str


RiskLevel = Literal["low", "medium", "high"]


class LegalGraphState(TypedDict, total=False):
    # ===== Metadata =====
    request_id: str

    # ===== Input =====
    question: str

    # ===== Interpretação =====
    domain: str
    legal_intent: str
    missing_information: List[str]

    # ===== Retrieval =====
    queries: List[str]
    documents: List[LegalDocument]

    # ===== Cross-reference =====
    has_conflict: bool
    conflicts: List[str]

    # ===== Risk analysis =====
    risk_level: RiskLevel
    risk_factors: List[str]
    recommendation: str

    # ===== Output =====
    answer: str
    disclaimer: str