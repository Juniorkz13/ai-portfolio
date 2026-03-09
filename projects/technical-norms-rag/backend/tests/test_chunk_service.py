from app.services.chunk_service import ChunkService


def test_create_chunks_page_with_text():
    service = ChunkService()
    pages = [{"page_number": 1, "text": "Primeira página com conteúdo técnico."}]

    chunks = service.create_chunks(pages, chunk_size=100, chunk_overlap=20)

    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["chunk_index"] == 0
    assert "conteúdo técnico" in chunks[0]["content"]


def test_create_chunks_ignores_empty_page():
    service = ChunkService()
    pages = [{"page_number": 2, "text": "   \n\t  "}]

    chunks = service.create_chunks(pages, chunk_size=100, chunk_overlap=20)

    assert chunks == []


def test_create_chunks_generates_multiple_chunks_and_sequential_indexes():
    service = ChunkService()
    text = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
        "nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
    )
    pages = [{"page_number": 10, "text": text}]

    chunks = service.create_chunks(pages, chunk_size=40, chunk_overlap=10)

    assert len(chunks) > 1
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk["page_number"] == 10 for chunk in chunks)
    assert "zeta eta" in chunks[0]["content"]
    assert "zeta eta" in chunks[1]["content"]
