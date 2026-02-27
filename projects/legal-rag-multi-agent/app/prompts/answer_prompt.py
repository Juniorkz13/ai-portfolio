from app.core.state import LegalGraphState

def build_answer_prompt(state: LegalGraphState) -> str:
    return f"""
Pergunta original:
{state['question']}

Domínio jurídico:
{state['domain']}

Intenção:
{state['legal_intent']}

Documentos relevantes:
{state['documents']}

Conflitos detectados:
{state.get('conflicts', [])}

Nível de risco:
{state['risk_level']}

Fatores de risco:
{state['risk_factors']}

Recomendação:
{state['recommendation']}

Gere uma resposta clara, objetiva e informativa.
Não forneça aconselhamento jurídico definitivo.
"""