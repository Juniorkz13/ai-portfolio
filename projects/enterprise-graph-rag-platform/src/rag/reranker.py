from typing import List, Dict
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Reranks retrieved documents using a CrossEncoder.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 3,
    ):
        self.model = CrossEncoder(model_name)
        self.top_n = top_n

    def rerank(self, query: str, docs: List[Dict]) -> List[Dict]:
        """
        Rerank documents based on relevance to the query.
        """
        if not docs:
            return docs

        pairs = [
            (query, doc["metadata"]["content"])
            for doc in docs
        ]

        scores = self.model.predict(pairs)

        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(
            docs,
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return reranked[: self.top_n]
