from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.database import Base
from app.models.chunk import Chunk
from app.models.document import Document


def _build_session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return Session(bind=engine)


def test_document_chunk_relationship():
    session = _build_session()

    document = Document(
        title="Norma de Seguranca",
        file_path="storage/pdfs/norma.pdf",
        document_type="regulation",
        version="1.0",
    )
    document.chunks.append(
        Chunk(
            page_number=5,
            content="Largura minima da escada de emergencia.",
            embedding=[0.12, 0.34, 0.56],
        )
    )

    session.add(document)
    session.commit()

    persisted_document = session.scalar(select(Document))
    assert persisted_document is not None
    assert len(persisted_document.chunks) == 1
    assert persisted_document.chunks[0].page_number == 5
    assert persisted_document.chunks[0].embedding == [0.12, 0.34, 0.56]

    session.close()


def test_delete_document_cascades_chunks():
    session = _build_session()

    document = Document(
        title="Codigo de Obras",
        file_path="storage/pdfs/codigo.pdf",
        document_type="code",
        version="2026",
    )
    document.chunks.append(
        Chunk(
            page_number=2,
            content="Texto tecnico.",
            embedding=None,
        )
    )

    session.add(document)
    session.commit()

    session.delete(document)
    session.commit()

    remaining_chunks = session.scalars(select(Chunk)).all()
    assert remaining_chunks == []

    session.close()
