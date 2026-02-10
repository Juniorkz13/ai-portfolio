from pathlib import Path
from typing import List

from pathlib import Path

from src.rag.loader import load_documents_from_dir
from src.rag.splitter import split_text
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.llm.model import LocalLLM


class RAGPipeline:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.embedding_model = EmbeddingModel()

        import faiss
        import numpy as np

        dimension = self.embedding_model.dimension
        index = faiss.IndexFlatL2(dimension)

        self.vector_store = VectorStore(index)

        self.llm = LocalLLM()

    def ingest(self):
        documents = load_documents_from_dir(self.data_dir)

        if not documents:
            return

        chunks = []
        for doc in documents:
            chunks.extend(split_text(doc))

        embeddings = self.embedding_model.embed(chunks)
        self.vector_store.add_documents(chunks, embeddings)



    def _build_prompt(self, context: str, question: str) -> str:
        return f"""
Você é um sistema de perguntas e respostas baseado em recuperação (RAG).

REGRAS OBRIGATÓRIAS:
- Use SOMENTE frases que aparecem no contexto.
- NÃO explique.
- NÃO acrescente informações.
- NÃO reescreva com suas próprias palavras.
- NÃO use conhecimento externo.
- NÃO utilize pronomes pessoais como "eu", "nós", "meu", "nosso", etc.
- Responda sempre em terceira pessoa, referindo-se ao conteúdo do contexto.
- Se a resposta não estiver literalmente no contexto, responda exatamente:
"Não sei responder com base nos documentos disponíveis."

CONTEXTO:
{context}

PERGUNTA:
{question}

RESPOSTA:
"""



    def ask(self, question: str) -> str:
        if self.vector_store.is_empty():
            return (
                "Nenhum documento foi indexado ainda. "
                "Adicione arquivos em data/raw e execute /ingest."
            )
        
        query_embedding = self.embedding_model.embed([question])[0]

        top_k = 5
        retrieved_docs = self.vector_store.query(
            query_embedding,
            top_k=top_k
        )

        print(type(retrieved_docs))
        print(retrieved_docs)

        if retrieved_docs is None or len(retrieved_docs) == 0:
            return "Não sei responder com base nos documentos disponíveis."
        
        unique_docs = list(dict.fromkeys(retrieved_docs))

        context = "\n".join(unique_docs).strip()

        print("\n================ CONTEXT USED ================\n")
        print(context)
        print("\n==============================================\n")

        if not context:
            return "Não sei responder com base nos documentos disponíveis."
        
        prompt = self._build_prompt(context, question)
        answer = self.llm.generate(prompt)
        return answer



