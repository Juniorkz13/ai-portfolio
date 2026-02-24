from pathlib import Path
from typing import List
from pypdf import PdfReader
import re


PDF_DIR = Path("data/pdfs")


def load_pdfs() -> str:
    """
    Lê todos os PDFs da pasta data/pdfs e retorna o texto consolidado.
    """
    all_text = []

    for pdf_file in PDF_DIR.glob("*.pdf"):
        reader = PdfReader(pdf_file)

        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)

    return "\n".join(all_text)


def clean_text(text: str) -> str:
    """
    Remove espaços extras, quebras desnecessárias e normaliza o texto.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[str]:
    """
    Divide o texto em chunks com overlap.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def ingest_documents() -> List[str]:
    """
    Pipeline completo de ingestão:
    PDF -> texto -> limpeza -> chunks
    """
    raw_text = load_pdfs()
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text)

    return chunks