from typing import List

def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start = end - overlap

    return chunks
