from app.llm import generate_answer
from app.prompts import build_prompt
from app.ingest import ingest_documents
from app.embeddings import embed_texts
from app.vectorstore import load_index, search_index


def answer_question(question: str, top_k: int = 3) -> str:
    chunks = ingest_documents()
    query_embedding = embed_texts([question])[0]
    index = load_index()

    top_indices = search_index(index, query_embedding, top_k)
    relevant_chunks = [chunks[i] for i in top_indices]
    context = "\n\n".join(relevant_chunks)

    prompt = build_prompt(context, question)
    answer = generate_answer(prompt)

    return answer