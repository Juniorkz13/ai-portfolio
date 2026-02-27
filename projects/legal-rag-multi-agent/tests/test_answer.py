from app.agents.answer import AnswerAgent
from app.llm.client import FakeLLMClient


def test_answer_agent_returns_answer_and_disclaimer():
    agent = AnswerAgent(FakeLLMClient())

    result = agent.run({
        "question": "Posso demitir por justa causa?",
        "documents": [],
        "risk_level": "medio",
        "risk_factors": ["Informações incompletas"],
        "recommendation": "Avaliação jurídica recomendada"
    })

    assert "answer" in result
    assert "disclaimer" in result