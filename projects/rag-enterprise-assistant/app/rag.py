from app.embeddings import embed_texts
from app.vectorstore import search_index
from app.prompts import build_prompt
from app.llm import generate_answer
from app import state
from app.memory import get_history, add_message
from functools import lru_cache

from app.config import get_settings

settings = get_settings()

TOP_K = settings["rag_top_k"]


def _answer_pipeline(question: str, session_id: str, top_k: int = 3) -> str:
    query_embedding = embed_texts([question])[0]

    top_indices = search_index(
        state.faiss_index,
        query_embedding,
        TOP_K
    )

    relevant_chunks = [state.chunks[i] for i in top_indices]
    context = "\n\n".join(relevant_chunks)

    history = get_history(session_id)

    prompt = build_prompt(
        context=context,
        question=question,
        history=history
    )

    answer = generate_answer(prompt)

    add_message(session_id, question, answer)

    return answer


@lru_cache(maxsize=128)
def _cached_answer(question: str, session_id: str) -> str:
    return _answer_pipeline(question, session_id)


def answer_question(question: str, session_id: str) -> str:
    if state.chunks is None or state.faiss_index is None:
        raise RuntimeError("RAG pipeline não inicializado")

    normalized = question.strip().lower()
    return _cached_answer(normalized, session_id)