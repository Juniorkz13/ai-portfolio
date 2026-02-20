from typing import List, Dict


class TextSplitter:
    """
    Responsible for splitting large documents into semantically
    coherent chunks suitable for embedding.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[Dict]:
        """
        Split text into overlapping chunks.

        Returns:
            List of dicts with:
            - content: chunk text
            - chunk_index: position in document
        """
        words = text.split()
        chunks = []

        start = 0
        chunk_index = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]

            chunk_text = " ".join(chunk_words)

            chunks.append(
                {
                    "content": chunk_text,
                    "chunk_index": chunk_index,
                }
            )

            chunk_index += 1
            start += self.chunk_size - self.chunk_overlap

        return chunks
