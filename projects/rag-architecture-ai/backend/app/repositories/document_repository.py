from sqlalchemy.orm import Session

from app.models.document import Document


class DocumentRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, *, title: str, file_path: str, document_type: str = "unknown", version: str = "1.0") -> Document:
        document = Document(
            title=title,
            file_path=file_path,
            document_type=document_type,
            version=version,
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document
