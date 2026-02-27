from app.agents.risk import RiskAgent


def test_high_risk_when_conflict_and_missing_info():
    agent = RiskAgent()

    result = agent.run({
        "has_conflict": True,
        "conflicts": ["Divergência entre lei e jurisprudência"],
        "missing_information": ["provas", "tipo da falta"]
    })

    assert result["risk_level"] == "alto"


def test_low_risk_when_no_conflict():
    agent = RiskAgent()

    result = agent.run({
        "has_conflict": False,
        "conflicts": [],
        "missing_information": []
    })

    assert result["risk_level"] == "baixo"