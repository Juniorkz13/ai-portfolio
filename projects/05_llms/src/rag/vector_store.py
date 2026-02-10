class VectorStore:
    def __init__(self, index):
        self.index = index
        self.documents: list[str] = []
        self._size = 0

    def add_documents(self, documents, embeddings):
        self.index.add(embeddings)
        self.documents.extend(documents)
        self._size += len(documents)

    def query(self, query_embedding, top_k=5) -> list[str]:
        if self._size == 0:
            return []

        distances, indices = self.index.search(
            query_embedding.reshape(1, -1),
            top_k
        )

        return [self.documents[i] for i in indices[0] if i < len(self.documents)]

    def is_empty(self) -> bool:
        return self._size == 0
