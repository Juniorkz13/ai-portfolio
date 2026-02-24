from typing import List, Dict
import faiss
import numpy as np
import os
import pickle


class FaissVectorStore:
    """
    FAISS-based vector store with metadata support.
    Designed for enterprise-grade RAG systems.
    """

    def __init__(self, embedding_dim: int, index_path: str = "faiss.index"):
        self.embedding_dim = embedding_dim
        self.index_path = index_path

        # Inner Product index
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        # Metadata storage
        self.metadata: List[Dict] = []

    def add_documents(
        self,
        embeddings: np.ndarray,
        metadatas: List[Dict],
    ):
        """
        Add document embeddings and metadata to the index.
        """
        if len(embeddings) != len(metadatas):
            raise ValueError("Embeddings and metadata size mismatch")

        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Search for the most similar documents.
        """
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            result = {
                "score": float(score),
                "metadata": self.metadata[idx],
            }
            results.append(result)

        return results

    def save(self):
        """
        Persist FAISS index and metadata.
        """
        faiss.write_index(self.index, self.index_path)

        with open(f"{self.index_path}.meta", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        """
        Load FAISS index and metadata from disk.
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError("FAISS index not found")

        self.index = faiss.read_index(self.index_path)

        with open(f"{self.index_path}.meta", "rb") as f:
            self.metadata = pickle.load(f)
