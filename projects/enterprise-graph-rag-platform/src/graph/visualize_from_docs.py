from src.rag.vector_store import FaissVectorStore
from src.rag.embeddings import EmbeddingService
from src.rag.pipeline import RetrievalPipeline
from src.rag.reranker import CrossEncoderReranker

from src.graph.graph_builder import build_graph_from_docs
from src.graph.graph_visualizer import visualize_graph


def main():

    embedding_service = EmbeddingService()

    vector_store = FaissVectorStore(embedding_dim=384)
    vector_store.load()

    reranker = CrossEncoderReranker(top_n=3)

    pipeline = RetrievalPipeline(
        embedding_service=embedding_service,
        vector_store=vector_store,
        top_k=5,
        reranker=reranker,
    )

    entity = "José da Silva"
    docs = pipeline.retrieve(entity)

    graph = build_graph_from_docs(docs, root_entity=entity)

    visualize_graph(graph, output_path="graph.png")


if __name__ == "__main__":
    main()
