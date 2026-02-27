from app.agents.cross_reference import CrossReferenceAgent


def test_detects_conflict_between_law_and_jurisprudence():
    agent = CrossReferenceAgent()

    documents = [
        {
            "text": "Texto da CLT sobre justa causa",
            "source": "CLT Art. 482",
            "metadata": {},
        },
        {
            "text": "Decisão do TST flexibilizando justa causa",
            "source": "Jurisprudência TST",
            "metadata": {},
        },
    ]

    result = agent.run({"documents": documents})

    assert result["has_conflict"] is True
    assert len(result["conflicts"]) > 0


def test_no_conflict_when_single_source():
    agent = CrossReferenceAgent()

    documents = [
        {
            "text": "Texto da CLT sobre justa causa",
            "source": "CLT Art. 482",
            "metadata": {},
        }
    ]

    result = agent.run({"documents": documents})

    assert result["has_conflict"] is False