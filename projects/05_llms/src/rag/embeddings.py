from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, texts):
        return self.model.encode(texts)

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
