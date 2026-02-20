from src.rag.loader import DocumentLoader
from src.rag.splitter import TextSplitter
from src.rag.embeddings import EmbeddingService
from src.rag.vector_store import FaissVectorStore


def build_vector_index(
    data_path: str,
    index_path: str = "faiss.index",
):
    """
    One-time or batch job to build the vector index.
    """
    loader = DocumentLoader()
    splitter = TextSplitter()
    embedder = EmbeddingService()

    vector_store = FaissVectorStore(
        embedding_dim=384,
        index_path=index_path,
    )

    documents = loader.load(data_path)

    all_chunks = []
    all_metadata = []

    for doc in documents:
        chunks = splitter.split_text(doc["content"])

        for chunk in chunks:
            all_chunks.append(chunk["content"])
            all_metadata.append(
                {
                    **doc["metadata"],
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"],
                }
            )

    embeddings = embedder.embed_texts(all_chunks)

    vector_store.add_documents(
        embeddings=embeddings,
        metadatas=all_metadata,
    )

    vector_store.save()

    print(f"Vector index built with {len(all_chunks)} chunks.")


if __name__ == "__main__":
    """
    Example usage:
    python app.py
    """

    DATA_PATH = "data/documents"
    INDEX_PATH = "faiss.index"

    build_vector_index(
        data_path=DATA_PATH,
        index_path=INDEX_PATH,
    )
