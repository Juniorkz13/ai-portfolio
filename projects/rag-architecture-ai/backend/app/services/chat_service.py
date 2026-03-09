class ChatService:
    def answer(self, question: str, context_chunks: list[dict[str, str | int]]) -> dict[str, str]:
        # Placeholder: call Gemini with context and format answer with citations.
        _ = (question, context_chunks)
        return {
            "answer": "",
            "explanation": "",
            "sources": "",
        }
