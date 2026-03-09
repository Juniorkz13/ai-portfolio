from typing_extensions import TypedDict

from sqlalchemy.orm import Session

from app.models.chunk import Chunk


class ChunkCreateInput(TypedDict):
    """Payload required to create one chunk row."""

    page_number: int
    chunk_index: int
    content: str
    embedding: list[float] | None


class ChunkRepository:
    """Repository for persistence operations on `Chunk` entities."""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        *,
        document_id: int,
        page_number: int,
        chunk_index: int,
        content: str,
        embedding: list[float] | None,
    ) -> Chunk:
        """Persist a single chunk linked to a document."""
        chunk = Chunk(
            document_id=document_id,
            page_number=page_number,
            chunk_index=chunk_index,
            content=content,
            embedding=embedding,
        )
        try:
            self.db.add(chunk)
            self.db.commit()
            self.db.refresh(chunk)
        except Exception:
            self.db.rollback()
            raise
        return chunk

    def create_many(
        self,
        *,
        document_id: int,
        chunks: list[ChunkCreateInput],
    ) -> list[Chunk]:
        """Persist multiple chunks for a given document in a single transaction."""
        chunk_models: list[Chunk] = []
        try:
            for item in chunks:
                chunk_models.append(
                    Chunk(
                        document_id=document_id,
                        page_number=item["page_number"],
                        chunk_index=item["chunk_index"],
                        content=item["content"],
                        embedding=item["embedding"],
                    )
                )
            self.db.add_all(chunk_models)
            self.db.commit()
            for chunk in chunk_models:
                self.db.refresh(chunk)
        except Exception:
            self.db.rollback()
            raise
        return chunk_models
