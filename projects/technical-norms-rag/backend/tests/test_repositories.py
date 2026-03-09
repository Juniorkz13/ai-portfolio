from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.database import Base
from app.models.chunk import Chunk
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository


def _make_session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return Session(bind=engine)


def test_document_repository_create_and_get_by_id():
    session = _make_session()
    repository = DocumentRepository(session)

    created = repository.create(
        title="IT-01",
        file_path="storage/pdfs/it-01.pdf",
        document_type="fire_regulation",
        version="2026.1",
    )
    fetched = repository.get_by_id(created.id)

    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.title == "IT-01"

    session.close()


def test_chunk_repository_create_many_for_document():
    session = _make_session()
    document_repository = DocumentRepository(session)
    chunk_repository = ChunkRepository(session)

    document = document_repository.create(
        title="NBR Access",
        file_path="storage/pdfs/nbr-access.pdf",
    )

    saved_chunks = chunk_repository.create_many(
        document_id=document.id,
        chunks=[
            {
                "page_number": 3,
                "chunk_index": 0,
                "content": "Primeiro trecho tecnico.",
                "embedding": [0.1, 0.2, 0.3],
            },
            {
                "page_number": 4,
                "chunk_index": 1,
                "content": "Segundo trecho tecnico.",
                "embedding": [0.4, 0.5, 0.6],
            },
        ],
    )

    persisted = session.scalars(
        select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.id)
    ).all()

    assert len(saved_chunks) == 2
    assert len(persisted) == 2
    assert persisted[0].page_number == 3
    assert persisted[1].page_number == 4
    assert persisted[0].chunk_index == 0
    assert persisted[1].chunk_index == 1
    assert persisted[0].embedding == [0.1, 0.2, 0.3]
    assert persisted[1].embedding == [0.4, 0.5, 0.6]

    session.close()
