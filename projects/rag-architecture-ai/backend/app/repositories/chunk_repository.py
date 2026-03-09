from sqlalchemy.orm import Session

from app.models.chunk import Chunk


class ChunkRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, *, document_id: int, page_number: int, content: str, embedding: str) -> Chunk:
        chunk = Chunk(
            document_id=document_id,
            page_number=page_number,
            content=content,
            embedding=embedding,
        )
        self.db.add(chunk)
        self.db.commit()
        self.db.refresh(chunk)
        return chunk
