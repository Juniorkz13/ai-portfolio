from app.agents.query_planner import QueryPlannerAgent

def test_generates_queries_for_labor_law():
    agent = QueryPlannerAgent()

    input_data = {
        "domain": "Direito do Trabalho",
        "legal_intent": "Verificar possibilidade de justa causa",
        "missing_information": []
    }

    result = agent.run(input_data)

    assert "queries" in result
    assert len(result["queries"]) >= 2
    assert any("CLT" in q for q in result["queries"])