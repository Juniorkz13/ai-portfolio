# src/evals/retrieval_eval.py

import json
from typing import List
from src.rag.pipeline import RetrievalPipeline


def precision_at_k(retrieved: List[str], expected: str, k: int) -> float:
    return 1.0 if any(expected in r for r in retrieved[:k]) else 0.0


def recall_at_k(retrieved: List[str], expected: str) -> float:
    return 1.0 if any(expected in r for r in retrieved) else 0.0


def evaluate_pipeline(
    pipeline: RetrievalPipeline,
    dataset_path: str,
    k: int = 3,
):
    with open(dataset_path) as f:
        dataset = json.load(f)

    precision_scores = []
    recall_scores = []

    for item in dataset:
        query = item["query"]
        expected = item["expected_answer"]

        docs = pipeline.retrieve(query)
        contents = [d["metadata"]["content"] for d in docs]

        p = precision_at_k(contents, expected, k)
        r = recall_at_k(contents, expected)

        precision_scores.append(p)
        recall_scores.append(r)

        print(f"\nQuery: {query}")
        print(f"Expected: {expected}")
        print(f"Retrieved: {contents[:2]}")
        print(f"Precision@{k}: {p} | Recall: {r}")

    print("\n==== FINAL METRICS ====")
    print(f"Precision@{k}: {sum(precision_scores)/len(precision_scores):.2f}")
    print(f"Recall@{k}: {sum(recall_scores)/len(recall_scores):.2f}")
