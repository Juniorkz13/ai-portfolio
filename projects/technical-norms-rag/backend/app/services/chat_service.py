import logging
import re
from typing import Protocol
from typing_extensions import TypedDict

from app.core.config import settings
from app.services.retrieval_service import RetrievalFilters


class RetrievedChunk(TypedDict):
    """Structured chunk returned by retrieval service."""

    content: str
    document_title: str
    document_type: str
    version: str
    page_number: int
    chunk_index: int
    document_id: int


class SourceReference(TypedDict):
    """Minimal source payload returned with chat answer."""

    document_id: int
    document_title: str
    document_type: str
    version: str
    page_number: int
    chunk_index: int
    excerpt: str


class ChatResponse(TypedDict):
    """Response schema for chat answers grounded in retrieved context."""

    answer: str
    explanation: str
    sources: list[SourceReference]


class RetrievalClient(Protocol):
    """Contract required from retrieval service dependency."""

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        """Return the most relevant chunks for a given question."""


class LLMClient(Protocol):
    """Contract required from LLM client dependency."""

    def generate(self, prompt: str, model: str | None = None) -> str:
        """Generate text completion from a fully rendered prompt."""


class ChatServiceError(Exception):
    """Raised when chat orchestration fails."""


class ChatService:
    """Compose retrieval context + LLM generation for grounded QA."""

    def __init__(
        self,
        retrieval_service: RetrievalClient,
        llm_client: LLMClient,
        *,
        model_name: str = settings.gemini_model,
    ):
        self.logger = logging.getLogger(__name__)
        self.retrieval_service = retrieval_service
        self.llm_client = llm_client
        self.model_name = "gemini-flash-latest"
        if model_name != self.model_name:
            # Enforce single allowed model across the application.
            self.logger.warning(
                "Ignoring unsupported model override; enforcing gemini-flash-latest",
                extra={"requested_model": model_name},
            )

    def answer(
        self,
        question: str,
        top_k: int = 5,
        filters: RetrievalFilters | None = None,
    ) -> ChatResponse:
        """Answer a user question using only retrieved document context."""
        self.logger.info(
            "Chat request received",
            extra={"question_length": len(question or ""), "top_k": top_k, "has_filters": bool(filters)},
        )
        if not question or not question.strip():
            raise ValueError("question must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        try:
            self.logger.info("Retrieving relevant chunks")
            chunks = self.retrieval_service.retrieve(question, top_k=top_k, filters=filters)
            self.logger.info("Chunks retrieved", extra={"retrieved_chunks": len(chunks)})
        except Exception as exc:
            self.logger.exception("Failed during chunk retrieval step")
            raise ChatServiceError("Failed to retrieve context for chat answer.") from exc

        if not self._has_sufficient_context(chunks):
            self.logger.info("Insufficient context for safe answer", extra={"retrieved_chunks": len(chunks)})
            return self._insufficient_context_response()

        self.logger.info("Building prompt from retrieved context")
        prompt = self._build_prompt(question, chunks)
        self.logger.info("Prompt built", extra={"prompt_length": len(prompt), "model_name": self.model_name})
        try:
            self.logger.info("Calling Gemini model")
            llm_text = self.llm_client.generate(prompt=prompt, model=self.model_name)
            self.logger.info("Gemini response received", extra={"response_length": len(llm_text or "")})
        except Exception as exc:
            self.logger.exception("Failed during Gemini call")
            raise ChatServiceError("Failed to generate answer with the configured LLM.") from exc

        self.logger.info("Formatting chat response")
        answer_text, explanation_text = self._parse_llm_response(llm_text)
        self.logger.info("Chat response formatted successfully")
        return {
            "answer": answer_text,
            "explanation": explanation_text,
            "sources": self._build_sources(chunks),
        }

    def _has_sufficient_context(self, chunks: list[RetrievedChunk]) -> bool:
        """Check whether retrieved context has minimal signal for grounded answer."""
        if not chunks:
            return False
        combined_length = sum(len(chunk["content"].strip()) for chunk in chunks)
        return combined_length >= 40

    def _build_prompt(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Build a natural, context-grounded prompt for concise professional answers."""
        context_lines = []
        for chunk in chunks:
            context_lines.append(
                (
                    f"[document_id={chunk['document_id']} "
                    f"title={chunk['document_title']} "
                    f"type={chunk['document_type']} "
                    f"version={chunk['version']} "
                    f"page={chunk['page_number']} chunk={chunk['chunk_index']}] "
                    f"{chunk['content']}"
                )
            )
        context_block = "\n".join(context_lines)

        return (
            "Você é um assistente técnico especializado em normas de arquitetura e segurança.\n"
            "Responda em português natural, claro e profissional, como um especialista explicando rapidamente o conteúdo.\n"
            "Regras obrigatórias:\n"
            "1. Use apenas o contexto fornecido.\n"
            "2. Não invente, não extrapole e não suponha informações fora dos trechos.\n"
            "3. Se o contexto for insuficiente, diga isso explicitamente.\n"
            "4. Não use títulos artificiais como 'Resposta objetiva' ou 'Explicação técnica'.\n"
            "5. Escreva entre 1 e 3 parágrafos, com síntese útil para leitura humana.\n"
            "6. Quando relevante, cite o nome/tipo/versão do documento de forma natural no texto.\n"
            "7. Evite copiar frases longas literalmente; prefira síntese fiel.\n\n"
            "Ajuste de estilo conforme a pergunta:\n"
            "- Pergunta geral/visão geral: responda de forma resumida, fluida e explicativa.\n"
            "- Pergunta técnica específica: responda com precisão técnica e linguagem clara.\n"
            "- Evite bullets, listas e blocos fragmentados, salvo se estritamente necessário.\n\n"
            "Formato de saída obrigatório:\n"
            "- Retorne apenas o texto final da resposta, sem rótulos, sem seções e sem títulos.\n"
            "- Não escreva expressões como 'Resposta objetiva', 'Explicação técnica' ou 'Resultado da consulta'.\n\n"
            f"Pergunta:\n{question}\n\n"
            f"Contexto recuperado:\n{context_block}\n\n"
            "Agora redija apenas a resposta final para o usuário."
        )

    def _parse_llm_response(self, response: str) -> tuple[str, str]:
        """Normalize LLM output into a natural answer and concise explanation."""
        text = response.strip()
        if not text:
            return (
                "Não foi possível gerar uma resposta confiável com o contexto atual.",
                "Não foi possível sintetizar uma resposta útil a partir dos trechos recuperados.",
            )

        # Guard against legacy/formalized outputs by stripping artificial labels and headings.
        cleaned_lines: list[str] = []
        for line in text.splitlines():
            normalized = line.strip()
            if self._is_artificial_heading(normalized):
                continue
            normalized = self._strip_known_prefix(normalized)
            normalized = self._strip_list_marker(normalized)
            if normalized:
                cleaned_lines.append(normalized)

        answer = self._sanitize_answer_text(self._compact_answer_lines(cleaned_lines) or text)
        explanation = "Resposta sintetizada a partir dos trechos mais relevantes recuperados nos documentos."
        return answer, explanation

    def _strip_known_prefix(self, line: str) -> str:
        """Remove legacy field prefixes in pt/en while preserving content."""
        lowered = line.lower()
        prefixes = (
            "answer:",
            "resposta:",
            "explanation:",
            "explicação:",
            "explicacao:",
            "observations:",
            "observações:",
            "observacoes:",
            "resposta objetiva:",
            "explicação técnica:",
            "explicacao tecnica:",
        )
        for prefix in prefixes:
            if lowered.startswith(prefix):
                return line[len(prefix) :].strip()
        return line

    def _is_artificial_heading(self, line: str) -> bool:
        """Detect artificial section headings that should not appear in final answer."""
        normalized = line.strip().lower().strip("*#:- ")
        artificial_headings = {
            "resposta objetiva",
            "explicação técnica",
            "explicacao tecnica",
            "observações importantes",
            "observacoes importantes",
            "resultado da consulta",
            "explanation",
            "answer",
            "observations",
        }
        return normalized in artificial_headings

    def _strip_list_marker(self, line: str) -> str:
        """Remove simple list markers to keep natural paragraph style."""
        return re.sub(r"^[-*]\s+", "", line).strip()

    def _compact_answer_lines(self, lines: list[str]) -> str:
        """Join fragmented lines while preserving paragraph breaks."""
        if not lines:
            return ""
        paragraphs: list[list[str]] = [[]]
        for line in lines:
            if not line:
                if paragraphs[-1]:
                    paragraphs.append([])
                continue
            paragraphs[-1].append(line)
        joined_paragraphs = [" ".join(chunk).strip() for chunk in paragraphs if chunk]
        return "\n\n".join(joined_paragraphs).strip()

    def _sanitize_answer_text(self, text: str) -> str:
        """Remove residual section labels if they appear inline in model output."""
        patterns = (
            r"(?im)^\s*(?:[#>*-]+\s*)?(?:\d+[\.\)]\s*)?(?:resposta\s+objetiva|explica(?:ç|c)ão\s+técnica|observa(?:ç|c)ões\s+importantes|resultado\s+da\s+consulta)\s*[:\-–]\s*",
            r"(?im)^\s*(?:[#>*-]+\s*)?(?:answer|explanation|observations)\s*[:\-–]\s*",
        )
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned)
        return cleaned.strip()

    def _build_sources(self, chunks: list[RetrievedChunk]) -> list[SourceReference]:
        """Map retrieved chunks into source references for traceability."""
        return [
            {
                "document_id": chunk["document_id"],
                "document_title": chunk["document_title"],
                "document_type": chunk["document_type"],
                "version": chunk["version"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "excerpt": self._build_excerpt(chunk["content"]),
            }
            for chunk in chunks
        ]

    def _build_excerpt(self, content: str, max_length: int = 180) -> str:
        """Build a short and safe excerpt from retrieved chunk content."""
        normalized = " ".join(content.split())
        if len(normalized) <= max_length:
            return normalized
        return f"{normalized[:max_length].rstrip()}..."

    def _insufficient_context_response(self) -> ChatResponse:
        """Return safe fallback when there is no reliable context to answer."""
        return {
            "answer": "Não foi possível responder com segurança com base no contexto disponível.",
            "explanation": (
                "Os trechos recuperados não trazem informação técnica suficiente para "
                "responder sem risco de inventar conteúdo normativo."
            ),
            "sources": [],
        }
