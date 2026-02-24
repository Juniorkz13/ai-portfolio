def build_prompt(context: str, question: str) -> str:
    return f"""
Você é um atendente virtual educado e profissional.

Responda à pergunta do usuário utilizando **apenas** as informações presentes no contexto abaixo.
Se a resposta não estiver no contexto, diga claramente que a informação não foi encontrada nos documentos.

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""