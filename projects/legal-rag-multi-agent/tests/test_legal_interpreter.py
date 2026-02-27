from app.agents.legal_interpreter import LegalInterpreterAgent

def test_detects_ambiguous_question():
    agent = LegalInterpreterAgent()
    result = agent.run({"question": "Posso demitir por justa causa?"})

    assert result["domain"] == "Direito do Trabalho"
    assert result["is_ambiguous"] is True
    assert len(result["missing_information"]) > 0