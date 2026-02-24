from app.ingest import ingest_documents
from app.embeddings import embed_texts
from app.vectorstore import load_index, build_faiss_index, save_index
from app import state


def init_rag():
    print("🔄 Inicializando RAG pipeline...")

    state.chunks = ingest_documents()

    try:
        state.faiss_index = load_index()
        print("✅ FAISS index carregado do disco")

    except FileNotFoundError:
        print("⚠️ FAISS index não encontrado. Criando novo índice...")

        embeddings = embed_texts(state.chunks)
        index = build_faiss_index(embeddings)
        save_index(index)

        state.faiss_index = index
        print("✅ FAISS index criado e salvo")

    print(f"✅ RAG pronto | chunks: {len(state.chunks)}")