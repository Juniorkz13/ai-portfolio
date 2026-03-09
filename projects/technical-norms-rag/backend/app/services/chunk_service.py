from typing_extensions import TypedDict


class PDFPage(TypedDict):
    page_number: int
    text: str


class TextChunk(TypedDict):
    content: str
    page_number: int
    chunk_index: int


class ChunkService:
    """Split extracted PDF page text into indexed chunks for retrieval."""

    def create_chunks(
        self,
        pages: list[PDFPage],
        chunk_size: int = 800,
        chunk_overlap: int | None = None,
    ) -> list[TextChunk]:
        """Build sequential chunks from structured PDF pages.

        Args:
            pages: Structured pages from `PDFService.extract_text_by_page`.
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of trailing characters repeated in next chunk.
                If None, a default of 20% of `chunk_size` is used.

        Returns:
            A list of chunks preserving page number and global sequential index.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if chunk_overlap is None:
            chunk_overlap = max(1, chunk_size // 5)
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        chunks: list[TextChunk] = []
        chunk_index = 0

        for page in pages:
            page_number = page["page_number"]
            text = page["text"]
            if not self._is_useful_text(text):
                continue

            for content in self._split_text(text, chunk_size, chunk_overlap):
                chunks.append(
                    {
                        "content": content,
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1

        return chunks

    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """Split text with a size-and-overlap sliding window."""
        normalized = " ".join(text.split())
        if not normalized:
            return []

        chunks: list[str] = []
        start = 0
        step = chunk_size - chunk_overlap

        while start < len(normalized):
            end = min(start + chunk_size, len(normalized))
            content = normalized[start:end].strip()
            if content:
                chunks.append(content)
            if end == len(normalized):
                break
            start += step

        return chunks

    def _is_useful_text(self, text: str) -> bool:
        """Return True if text contains meaningful alphanumeric content."""
        if not text or not text.strip():
            return False
        return any(char.isalnum() for char in text)
