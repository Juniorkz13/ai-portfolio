from unittest.mock import patch
from app.agents.answer_agent import AnswerAgent

@patch("app.agents.answer_agent.generate_text")
@patch("app.agents.answer_agent.build_answer_prompt")
def test_answer_agent_run(mock_prompt, mock_llm):
    mock_prompt.return_value = "prompt gerado"
    mock_llm.return_value = "Resposta jurídica simulada"

    state = {
        "question": "Pergunta teste",
        "domain": "Direito Civil",
        "legal_intent": "Teste",
        "documents": [],
        "risk_level": "low",
        "risk_factors": [],
        "recommendation": "Nenhuma",
    }

    result = AnswerAgent().run(state)

    assert "answer" in result
    assert "disclaimer" in result
    assert "Resposta jurídica simulada" in result["answer"]

    mock_prompt.assert_called_once_with(state)
    mock_llm.assert_called_once_with("prompt gerado")