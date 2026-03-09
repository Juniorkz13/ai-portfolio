from unittest.mock import Mock

import pytest

from app.services.chat_service import ChatService, ChatServiceError


def test_answer_returns_structured_response_from_context():
    retrieval = Mock()
    retrieval.retrieve.return_value = [
        {
            "content": "A largura mínima da escada de emergência é 1,20 m em edifícios de uso coletivo.",
            "document_title": "IT-01",
            "document_type": "fire_regulation",
            "version": "2026",
            "page_number": 14,
            "chunk_index": 2,
            "document_id": 10,
        }
    ]

    llm = Mock()
    llm.generate.return_value = (
        "A largura mínima indicada no trecho é de 1,20 m para edifícios de uso coletivo. "
        "Essa exigência busca garantir condições adequadas de evacuação em emergências."
    )

    service = ChatService(retrieval_service=retrieval, llm_client=llm)
    result = service.answer("Qual a largura mínima da escada?", top_k=3)

    assert "1,20 m" in result["answer"]
    assert "evacuação" in result["answer"]
    assert result["explanation"] == "Resposta sintetizada a partir dos trechos mais relevantes recuperados nos documentos."
    assert result["sources"] == [
        {
            "document_id": 10,
            "document_title": "IT-01",
            "document_type": "fire_regulation",
            "version": "2026",
            "page_number": 14,
            "chunk_index": 2,
            "excerpt": "A largura mínima da escada de emergência é 1,20 m em edifícios de uso coletivo.",
        }
    ]


def test_answer_returns_safe_message_when_context_is_insufficient():
    retrieval = Mock()
    retrieval.retrieve.return_value = []

    llm = Mock()

    service = ChatService(retrieval_service=retrieval, llm_client=llm)
    result = service.answer("Pergunta sem base")

    assert "não foi possível responder com segurança" in result["answer"].lower()
    assert result["sources"] == []
    llm.generate.assert_not_called()


def test_answer_wraps_retrieval_errors():
    retrieval = Mock()
    retrieval.retrieve.side_effect = RuntimeError("db failure")

    service = ChatService(retrieval_service=retrieval, llm_client=Mock())

    with pytest.raises(ChatServiceError, match="retrieve context"):
        service.answer("Pergunta")


def test_answer_wraps_llm_errors():
    retrieval = Mock()
    retrieval.retrieve.return_value = [
        {
            "content": "Conteúdo técnico suficiente para resposta confiável.",
            "document_title": "NBR",
            "document_type": "norm",
            "version": "1.0",
            "page_number": 3,
            "chunk_index": 0,
            "document_id": 1,
        }
    ]

    llm = Mock()
    llm.generate.side_effect = RuntimeError("llm offline")

    service = ChatService(retrieval_service=retrieval, llm_client=llm)

    with pytest.raises(ChatServiceError, match="configured LLM"):
        service.answer("Pergunta")


def test_answer_enforces_gemini_flash_latest_model():
    retrieval = Mock()
    retrieval.retrieve.return_value = [
        {
            "content": "Conteúdo técnico suficiente para resposta confiável e baseada em norma.",
            "document_title": "NBR",
            "document_type": "norm",
            "version": "1.0",
            "page_number": 8,
            "chunk_index": 1,
            "document_id": 2,
        }
    ]

    llm = Mock()
    llm.generate.return_value = "Answer: OK\nExplanation: OK"

    service = ChatService(retrieval_service=retrieval, llm_client=llm, model_name="invalid-model")
    service.answer("Pergunta")

    assert llm.generate.call_args.kwargs["model"] == "gemini-flash-latest"


def test_answer_passes_filters_to_retrieval():
    retrieval = Mock()
    retrieval.retrieve.return_value = [
        {
            "content": "Regra técnica aplicável ao documento filtrado.",
            "document_title": "IT-01",
            "document_type": "fire_regulation",
            "version": "2026",
            "page_number": 10,
            "chunk_index": 1,
            "document_id": 7,
        }
    ]
    llm = Mock()
    llm.generate.return_value = "OK"

    service = ChatService(retrieval_service=retrieval, llm_client=llm)
    service.answer("Pergunta", filters={"document_type": "fire_regulation", "version": "2026"})

    assert retrieval.retrieve.call_args.kwargs["filters"] == {
        "document_type": "fire_regulation",
        "version": "2026",
    }


def test_answer_strips_legacy_prefixed_lines():
    retrieval = Mock()
    retrieval.retrieve.return_value = [
        {
            "content": "A exigência se aplica a edifícios de uso coletivo com rota de fuga interna.",
            "document_title": "IT-01",
            "document_type": "fire_regulation",
            "version": "2026",
            "page_number": 15,
            "chunk_index": 5,
            "document_id": 9,
        }
    ]

    llm = Mock()
    llm.generate.return_value = (
        "Answer: A exigência é aplicável no cenário descrito.\n"
        "Explanation: O trecho recuperado define a regra para rota de fuga interna."
    )

    service = ChatService(retrieval_service=retrieval, llm_client=llm)
    result = service.answer("Quando essa exigência se aplica?")

    assert "A exigência é aplicável no cenário descrito." in result["answer"]
    assert "rota de fuga interna" in result["answer"]
    assert result["explanation"] == "Resposta sintetizada a partir dos trechos mais relevantes recuperados nos documentos."


def test_answer_removes_artificial_headings_in_portuguese():
    retrieval = Mock()
    retrieval.retrieve.return_value = [
        {
            "content": "O documento apresenta regras para organização de trabalhos acadêmicos e índices.",
            "document_title": "ABNT",
            "document_type": "norm",
            "version": "2024",
            "page_number": 2,
            "chunk_index": 0,
            "document_id": 3,
        }
    ]

    llm = Mock()
    llm.generate.return_value = (
        "## Resposta objetiva\n"
        "Os documentos tratam da organização e apresentação de trabalhos acadêmicos.\n\n"
        "### Explicação técnica\n"
        "- A NBR 14724 define a estrutura do trabalho.\n"
        "- A NBR 6034 trata da elaboração de índices."
    )

    service = ChatService(retrieval_service=retrieval, llm_client=llm)
    result = service.answer("Sobre o que se tratam os documentos?")

    assert "resposta objetiva" not in result["answer"].lower()
    assert "explicação técnica" not in result["answer"].lower()
    assert "NBR 14724 define a estrutura do trabalho." in result["answer"]
    assert "NBR 6034 trata da elaboração de índices." in result["answer"]


def test_answer_removes_resultado_da_consulta_label():
    retrieval = Mock()
    retrieval.retrieve.return_value = [
        {
            "content": "As normas descrevem critérios de citação e sumário em documentos técnicos.",
            "document_title": "ABNT",
            "document_type": "norm",
            "version": "2023",
            "page_number": 7,
            "chunk_index": 1,
            "document_id": 4,
        }
    ]

    llm = Mock()
    llm.generate.return_value = (
        "Resultado da consulta: Os documentos tratam de padronização de citações e de organização do sumário."
    )

    service = ChatService(retrieval_service=retrieval, llm_client=llm)
    result = service.answer("Sobre o que se tratam os documentos?")

    assert "resultado da consulta" not in result["answer"].lower()
    assert "padronização de citações" in result["answer"].lower()
