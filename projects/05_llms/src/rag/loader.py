from typing import List, Dict
from pathlib import Path
from pypdf import PdfReader


class DocumentLoader:
    """
    Responsible for loading raw documents and extracting text
    along with metadata for downstream processing.
    """

    SUPPORTED_EXTENSIONS = [".pdf", ".txt"]

    def load(self, path: str) -> List[Dict]:
        """
        Load documents from a file or directory.
        """
        path = Path(path)

        if path.is_dir():
            documents = []
            for file in path.iterdir():
                if file.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    documents.extend(self._load_file(file))
            return documents

        elif path.is_file():
            return self._load_file(path)

        else:
            raise ValueError("Invalid path provided")

    def _load_file(self, file_path: Path) -> List[Dict]:
        """
        Load a single file and extract its content.
        """
        if file_path.suffix.lower() == ".pdf":
            return self._load_pdf(file_path)

        if file_path.suffix.lower() == ".txt":
            return self._load_txt(file_path)

        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    def _load_pdf(self, file_path: Path) -> List[Dict]:
        reader = PdfReader(file_path)
        documents = []

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            documents.append(
                {
                    "content": text,
                    "metadata": {
                        "source": str(file_path),
                        "page": page_number,
                        "type": "pdf",
                    },
                }
            )

        return documents

    def _load_txt(self, file_path: Path) -> List[Dict]:
        text = file_path.read_text(encoding="utf-8")

        return [
            {
                "content": text,
                "metadata": {
                    "source": str(file_path),
                    "type": "txt",
                },
            }
        ]
