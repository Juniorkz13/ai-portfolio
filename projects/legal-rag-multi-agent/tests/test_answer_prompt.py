from app.prompts.answer_prompt import build_answer_prompt

def test_build_answer_prompt_basic():
    state = {
        "question": "Posso rescindir contrato?",
        "domain": "Direito Civil",
        "legal_intent": "Rescisão contratual",
        "documents": [
            {
                "source": "fake_db",
                "title": "Contrato Civil",
                "content": "Cláusulas sobre rescisão"
            }
        ],
        "risk_level": "medium",
        "risk_factors": ["Multa contratual"],
        "recommendation": "Analisar cláusulas antes de rescindir",
        "conflicts": []
    }

    prompt = build_answer_prompt(state)

    assert isinstance(prompt, str)
    assert "Posso rescindir contrato?" in prompt
    assert "Direito Civil" in prompt
    assert "Rescisão contratual" in prompt
    assert "Multa contratual" in prompt