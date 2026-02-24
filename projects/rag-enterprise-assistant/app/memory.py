from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

conversation_memory: Dict[str, Deque[Tuple[str, str]]] = defaultdict(
    lambda: deque(maxlen=5) 
)


def add_message(session_id: str, user_msg: str, assistant_msg: str):
    conversation_memory[session_id].append((user_msg, assistant_msg))


def get_history(session_id: str) -> str:
    """
    Retorna histórico formatado para o prompt.
    """
    history = conversation_memory.get(session_id)
    if not history:
        return ""

    formatted = []
    for user, assistant in history:
        formatted.append(f"Usuário: {user}")
        formatted.append(f"Atendente: {assistant}")

    return "\n".join(formatted)