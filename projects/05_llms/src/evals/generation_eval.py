import json
from src.llm.model import ModelRouter
from src.rag.pipeline import RetrievalPipeline


def is_faithful(answer: str, context: str) -> bool:
    """
    Simple heuristic faithfulness check.
    """
    return all(
        sentence.strip() in context
        for sentence in answer.split(".")
        if sentence.strip()
    )


def evaluate_generation(pipeline: RetrievalPipeline, dataset_path: str):
    with open(dataset_path) as f:
        dataset = json.load(f)

    model = ModelRouter()

    faithful = []

    for item in dataset:
        query = item["query"]

        docs = pipeline.retrieve(query)
        context = pipeline.build_context(docs)

        if not context:
            faithful.append(False)
            continue

        prompt = f"""
Answer the question strictly using the context.

Context:
{context}

Question:
{query}

Answer:
"""

        answer = model.run(prompt)

        f_score = is_faithful(answer, context)
        faithful.append(f_score)

        print("\nQuery:", query)
        print("Answer:", answer)
        print("Faithful:", f_score)

    print("\nFaithfulness:", sum(faithful)/len(faithful))
