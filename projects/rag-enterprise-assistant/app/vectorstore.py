import faiss
import numpy as np
from pathlib import Path
from typing import List


INDEX_PATH = Path("data/faiss_index")
INDEX_PATH.mkdir(parents=True, exist_ok=True)

INDEX_FILE = INDEX_PATH / "index.faiss"


def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_index(index):
    faiss.write_index(index, str(INDEX_FILE))


def load_index():
    if not INDEX_FILE.exists():
        raise FileNotFoundError("FAISS index não encontrado.")
    return faiss.read_index(str(INDEX_FILE))


def search_index(
    index,
    query_embedding: np.ndarray,
    top_k: int = 3
) -> List[int]:
    distances, indices = index.search(
        query_embedding.reshape(1, -1),
        top_k
    )
    return indices[0].tolist()