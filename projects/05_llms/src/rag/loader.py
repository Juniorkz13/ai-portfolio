from pathlib import Path
from typing import List
from pypdf import PdfReader

def load_txt(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def load_pdf(file_path: Path) -> str:
    reader = PdfReader(file_path)
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text())

    return "\n".join(pages)

def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = " ".join(text.split())
    return text

from pathlib import Path
from typing import List


def load_documents_from_dir(directory: Path) -> List[str]:
    if not directory.exists():
        return []

    documents = []

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append(f.read())

    return documents

