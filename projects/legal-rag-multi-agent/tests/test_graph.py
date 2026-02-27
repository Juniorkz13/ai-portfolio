from app.core.graph import build_graph


def test_graph_flow_returns_documents():
    graph = build_graph()

    result = graph.invoke({
        "question": "Posso demitir um funcionário por justa causa?"
    })

    assert "documents" in result
    assert len(result["documents"]) > 0