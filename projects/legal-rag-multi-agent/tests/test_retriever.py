from app.agents.retriever import RetrieverAgent

def test_retriever_returns_documents():
    agent = RetrieverAgent()

    result = agent.run({
        "queries": [
            "CLT artigo 482 justa causa",
            "jurisprudência TST justa causa"
        ]
    })

    assert "documents" in result
    assert len(result["documents"]) == 2
    assert "text" in result["documents"][0]