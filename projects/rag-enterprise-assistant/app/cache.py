from functools import lru_cache
from app.config import get_settings

settings = get_settings()


def normalize_question(question: str) -> str:
    return question.strip().lower()


@lru_cache(maxsize=settings["cache_size"])
def _cached_answer(question: str, session_id: str) -> str:
    """
    Cache genérico para respostas.
    """
    normalized_question = normalize_question(question)
    return normalized_question