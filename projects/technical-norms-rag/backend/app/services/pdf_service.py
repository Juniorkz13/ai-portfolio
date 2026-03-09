from pathlib import Path

import fitz


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be parsed or text extraction fails."""


class PDFService:
    def extract_text_by_page(self, file_path: str) -> list[dict[str, str | int]]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError("Only .pdf files are supported.")

        extracted_pages: list[dict[str, str | int]] = []
        try:
            with fitz.open(path) as document:
                for index, page in enumerate(document, start=1):
                    text = page.get_text("text").strip()
                    extracted_pages.append({"page_number": index, "text": text})
        except (fitz.FileDataError, RuntimeError, ValueError) as exc:
            raise PDFExtractionError(f"Unable to extract text from PDF: {file_path}") from exc

        return extracted_pages
