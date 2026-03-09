from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.document import Document


class DocumentRepository:
    """Repository for CRUD operations on `Document` entities."""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        *,
        title: str,
        file_path: str,
        document_type: str = "unknown",
        version: str = "1.0",
    ) -> Document:
        """Persist a new document and return it with generated fields."""
        document = Document(
            title=title,
            file_path=file_path,
            document_type=document_type,
            version=version,
        )
        try:
            self.db.add(document)
            self.db.commit()
            self.db.refresh(document)
        except Exception:
            self.db.rollback()
            raise
        return document

    def get_by_id(self, document_id: int) -> Document | None:
        """Fetch a document by primary key or return `None`."""
        return self.db.get(Document, document_id)

    def get_by_id_with_chunks(self, document_id: int) -> Document | None:
        """Fetch a document and preload chunks to compute `total_chunks`."""
        stmt = (
            select(Document)
            .options(selectinload(Document.chunks))
            .where(Document.id == document_id)
        )
        return self.db.scalar(stmt)

    def list_with_chunks(self) -> list[Document]:
        """List all documents with chunks preloaded for aggregate metadata."""
        stmt = (
            select(Document)
            .options(selectinload(Document.chunks))
            .order_by(Document.uploaded_at.desc())
        )
        return self.db.scalars(stmt).all()

    def delete_by_id(self, document_id: int) -> Document | None:
        """Delete a document by id and return the deleted entity snapshot."""
        document = self.get_by_id_with_chunks(document_id)
        if document is None:
            return None
        try:
            self.db.delete(document)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        return document
