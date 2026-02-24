from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Centralized embedding service.
    
    Responsible for:
    - Loading embedding models
    - Generating embeddings
    - Applying L2 normalization
    - Abstracting model implementation details
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.device = device

        self.model = SentenceTransformer(model_name, device=device)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32,
        )

        if self.normalize:
            embeddings = self._l2_normalize(embeddings)

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
        )

        if self.normalize:
            embedding = embedding / np.linalg.norm(embedding)

        return embedding

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """
        Apply L2 normalization to embeddings.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms
