import time
from typing import List, Dict, Any
from app.core.logging import get_logger
from app.agents.base import BaseAgent

logger = get_logger(__name__)


class RetrieverAgent(BaseAgent):
    """Agente de recuperação de documentos."""

    def __init__(
        self, vector_store=None, top_k: int = 5, similarity_threshold: float = 0.7
    ):
        """
        Inicializa o RetrieverAgent.
        
        Args:
            vector_store: Instância do vector store (ex: FAISS, Chroma, Pinecone)
            top_k: Número máximo de documentos a recuperar
            similarity_threshold: Limiar mínimo de similaridade (0-1)
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def run(self, input_data: dict) -> dict:
        start_time = time.time()

        try:
            question = input_data.get("question", "")
            request_id = input_data.get("request_id")
            top_k = input_data.get("top_k", self.top_k)

            # Recuperação de documentos
            documents = self._retrieve_documents(question, top_k)

            # Formata documentos para saída
            formatted_documents = [
                {
                    "text": doc.get("content") or doc.get("text", ""),
                    "source": doc.get("metadata", {}).get("source", "unknown"),
                    "metadata": {
                        k: v for k, v in doc.get("metadata", {}).items() 
                        if k != "source"
                    }
                }
                for doc in documents
            ]

            # Determina o método de recuperação usado
            retrieval_method = "vector_similarity" if self.vector_store else "keyword_fallback"

            result = {
                "documents": formatted_documents,
                "num_documents": len(formatted_documents),
                "retrieval_method": retrieval_method,
            }

            duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "documents_retrieved",
                extra={
                    "extra": {
                        "request_id": request_id,
                        "agent": "RetrieverAgent",
                        "num_documents": len(formatted_documents),
                        "retrieval_method": retrieval_method,
                        "top_k": top_k,
                        "duration_ms": duration_ms,
                    }
                },
            )

            return result

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            logger.error(
                "document_retrieval_failed",
                extra={
                    "extra": {
                        "request_id": input_data.get("request_id"),
                        "agent": "RetrieverAgent",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "duration_ms": duration_ms,
                    }
                },
                exc_info=True,
            )

            raise

    def _retrieve_documents(self, question: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Recupera documentos relevantes baseado na questão.
        
        Args:
            question: Pergunta do usuário
            top_k: Número de documentos a retornar (usa self.top_k se None)
            
        Returns:
            Lista de documentos com conteúdo e metadados
        """
        if top_k is None:
            top_k = self.top_k

        # Se temos um vector store configurado, usa busca por similaridade
        if self.vector_store:
            return self._vector_search(question, top_k)
        
        # Fallback para busca simulada baseada em keywords
        return self._keyword_based_search(question, top_k)

    def _vector_search(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Realiza busca por similaridade usando o vector store.
        
        Suporta diferentes tipos de vector stores:
        - LangChain VectorStore (FAISS, Chroma, Pinecone, etc.)
        - Vector stores com método similarity_search_with_score
        """
        try:
            # Tenta usar método padrão do LangChain
            if hasattr(self.vector_store, 'similarity_search_with_score'):
                # Retorna documentos com scores de similaridade
                results = self.vector_store.similarity_search_with_score(
                    question, 
                    k=top_k
                )
                
                documents = []
                for doc, score in results:
                    # Filtra por threshold de similaridade
                    if score >= self.similarity_threshold or score <= (1 - self.similarity_threshold):
                        documents.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "similarity_score": float(score)
                        })
                
                return documents
            
            # Fallback para similarity_search sem score
            elif hasattr(self.vector_store, 'similarity_search'):
                results = self.vector_store.similarity_search(question, k=top_k)
                
                documents = []
                for doc in results:
                    documents.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    })
                
                return documents
            
            # Para vector stores customizados com interface própria
            elif hasattr(self.vector_store, 'search'):
                results = self.vector_store.search(query=question, top_k=top_k)
                
                # Assume que retorna lista de dicts com 'content' e 'metadata'
                return results
            
            else:
                logger.warning(
                    "vector_store_method_not_found",
                    extra={
                        "extra": {
                            "vector_store_type": type(self.vector_store).__name__,
                        }
                    }
                )
                return self._keyword_based_search(question, top_k)
                
        except Exception as e:
            logger.error(
                "vector_search_failed",
                extra={
                    "extra": {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "vector_store_type": type(self.vector_store).__name__
                    }
                },
                exc_info=True
            )
            # Fallback para busca por keywords
            return self._keyword_based_search(question, top_k)

    def _keyword_based_search(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Busca simulada baseada em palavras-chave quando vector store não disponível.
        Útil para testes e desenvolvimento.
        """
        # Base de documentos simulados para diferentes tópicos jurídicos
        fake_documents = {
            "trabalhista": [
                {
                    "content": "CLT Art. 7º - São direitos dos trabalhadores urbanos e rurais, além de outros que visem à melhoria de sua condição social: I - relação de emprego protegida contra despedida arbitrária ou sem justa causa.",
                    "metadata": {"source": "CLT.pdf", "page": 12, "article": "Art. 7º", "area": "trabalhista"}
                },
                {
                    "content": "A rescisão do contrato de trabalho deve ser acompanhada do pagamento das verbas rescisórias devidas, incluindo saldo de salário, férias proporcionais, 13º salário proporcional e FGTS.",
                    "metadata": {"source": "manual_trabalhista.pdf", "page": 45, "area": "trabalhista"}
                },
                {
                    "content": "O aviso prévio é obrigatório quando uma das partes decide rescindir o contrato de trabalho sem justa causa. O prazo mínimo é de 30 dias, podendo ser acrescido de 3 dias por ano trabalhado.",
                    "metadata": {"source": "direito_trabalho.pdf", "page": 78, "area": "trabalhista"}
                }
            ],
            "civil": [
                {
                    "content": "Código Civil Art. 186 - Aquele que, por ação ou omissão voluntária, negligência ou imprudência, violar direito e causar dano a outrem, ainda que exclusivamente moral, comete ato ilícito.",
                    "metadata": {"source": "codigo_civil.pdf", "page": 89, "article": "Art. 186", "area": "civil"}
                },
                {
                    "content": "O contrato é um acordo de vontades entre duas ou mais partes para constituir, modificar ou extinguir relações jurídicas de natureza patrimonial.",
                    "metadata": {"source": "contratos.pdf", "page": 23, "area": "civil"}
                },
                {
                    "content": "A responsabilidade civil pode ser contratual ou extracontratual (aquiliana). Na primeira, decorre do descumprimento de obrigação previamente assumida.",
                    "metadata": {"source": "responsabilidade_civil.pdf", "page": 56, "area": "civil"}
                }
            ],
            "penal": [
                {
                    "content": "Código Penal Art. 121 - Matar alguém: Pena - reclusão, de seis a vinte anos. A tentativa de homicídio também é punível nos termos da lei.",
                    "metadata": {"source": "codigo_penal.pdf", "page": 34, "article": "Art. 121", "area": "penal"}
                },
                {
                    "content": "Considera-se em legítima defesa quem, usando moderadamente dos meios necessários, repele injusta agressão, atual ou iminente, a direito seu ou de outrem.",
                    "metadata": {"source": "direito_penal.pdf", "page": 67, "area": "penal"}
                },
                {
                    "content": "A prescrição penal extingue a punibilidade do crime após decorrido determinado prazo, que varia conforme a pena máxima do delito.",
                    "metadata": {"source": "prescricao_penal.pdf", "page": 12, "area": "penal"}
                }
            ],
            "consumidor": [
                {
                    "content": "CDC Art. 6º - São direitos básicos do consumidor: III - a informação adequada e clara sobre os diferentes produtos e serviços.",
                    "metadata": {"source": "CDC.pdf", "page": 8, "article": "Art. 6º", "area": "consumidor"}
                },
                {
                    "content": "O fornecedor de produtos ou serviços responde objetivamente pelos danos causados aos consumidores, independentemente da existência de culpa.",
                    "metadata": {"source": "defesa_consumidor.pdf", "page": 34, "area": "consumidor"}
                },
                {
                    "content": "O consumidor tem direito de arrepender-se da compra realizada fora do estabelecimento comercial no prazo de 7 dias (direito de arrependimento).",
                    "metadata": {"source": "direitos_consumidor.pdf", "page": 45, "area": "consumidor"}
                }
            ],
        }
        
        # Palavras-chave para identificar o tópico da questão
        keywords = {
            "trabalhista": ["trabalho", "emprego", "clt", "rescisão", "demissão", 
                           "férias", "salário", "justa causa", "aviso prévio", "fgts", 
                           "trabalhador", "empregado", "empregador", "contrato de trabalho"],
            "civil": ["contrato", "dano", "indenização", "responsabilidade civil", 
                     "ação civil", "obrigação", "patrimônio", "código civil", 
                     "ato ilícito", "reparação"],
            "penal": ["crime", "penal", "prisão", "reclusão", "homicídio", "roubo", 
                     "furto", "legítima defesa", "prescrição", "pena", "delito", 
                     "código penal"],
            "consumidor": ["consumidor", "produto", "serviço", "compra", "venda", 
                          "defeito", "garantia", "cdc", "fornecedor", "direito do consumidor"]
        }
        
        question_lower = question.lower()
        
        # Calcula score de relevância para cada tópico
        topic_scores = {}
        for topic, words in keywords.items():
            score = sum(1 for word in words if word in question_lower)
            if score > 0:
                topic_scores[topic] = score
        
        # Coleta documentos dos tópicos relevantes
        all_relevant_docs = []
        
        if topic_scores:
            # Ordena tópicos por relevância
            sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
            
            for topic, score in sorted_topics:
                docs = fake_documents.get(topic, [])
                # Adiciona score simulado aos documentos
                for doc in docs:
                    doc_copy = doc.copy()
                    doc_copy["metadata"] = doc["metadata"].copy()
                    # Score normalizado entre 0 e 1
                    doc_copy["similarity_score"] = min(0.5 + (score * 0.1), 0.95)
                    all_relevant_docs.append(doc_copy)
        else:
            # Se não houver match de keywords, retorna documentos genéricos
            default_docs = [
                {
                    "content": "A Constituição Federal de 1988 estabelece os direitos e garantias fundamentais dos cidadãos brasileiros, incluindo o direito à vida, liberdade, igualdade e segurança.",
                    "metadata": {"source": "constituicao.pdf", "page": 5, "area": "constitucional"},
                    "similarity_score": 0.3
                },
                {
                    "content": "O princípio da dignidade da pessoa humana é fundamento da República Federativa do Brasil e orienta toda a interpretação do ordenamento jurídico.",
                    "metadata": {"source": "principios_constitucionais.pdf", "page": 23, "area": "constitucional"},
                    "similarity_score": 0.3
                }
            ]
            all_relevant_docs = default_docs
        
        # Ordena por score e retorna top_k
        all_relevant_docs.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        return all_relevant_docs[:top_k]