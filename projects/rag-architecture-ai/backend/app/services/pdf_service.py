from pathlib import Path


class PDFService:
    def extract_text_by_page(self, file_path: str) -> list[dict[str, str | int]]:
        # Placeholder: integrate PyMuPDF or pypdf extraction here.
        _ = Path(file_path)
        return []
