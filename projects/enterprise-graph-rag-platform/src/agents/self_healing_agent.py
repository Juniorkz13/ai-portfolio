class SelfHealingAgent:
    """
    Automatic correction loop for RAG pipelines.
    """

    def heal(
        self,
        *,
        question: str,
        pipeline,
        model_router,
        review_agent,
        max_top_k: int = 10,
        task: str = "qa",
    ):

        pipeline.top_k = max_top_k
        retrieved_docs = pipeline.retrieve(question)

        if not retrieved_docs:
            return None, None, []

        context = pipeline.build_context(retrieved_docs)

        prompt = f"""
You are an AI assistant specialized in question answering.

Instructions:
- Answer strictly using the context.
- Be concise and factual.
- Do NOT speculate.

Context:
{context}

Question:
{question}

Answer:
"""

        answer = model_router.run(
            prompt=prompt,
            task=task,
        )

        review = review_agent.review(
            question=question,
            context=context,
            answer=answer,
        )

        return answer, review, retrieved_docs