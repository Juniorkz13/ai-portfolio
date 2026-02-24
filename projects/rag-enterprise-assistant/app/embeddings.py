from sentence_transformers import SentenceTransformer
import torch
from typing import List


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device=DEVICE
)

def embed_texts(texts: List[str]):
    """
    Gera embeddings para uma lista de textos.
    """
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )
    return embeddings