

def build_prompt(context: str, question: str, history: str = "") -> str:
    return f"""
Você é um atendente virtual educado e profissional.

Regras:
- Responda usando APENAS as informações do contexto.
- Use o histórico apenas para entender continuidade da conversa.
- Se a resposta não estiver no contexto, diga claramente que não encontrou a informação.

Histórico da conversa:
{history if history else "Nenhum histórico disponível."}

Contexto dos documentos:
{context}

Pergunta atual:
{question}

Resposta:
"""