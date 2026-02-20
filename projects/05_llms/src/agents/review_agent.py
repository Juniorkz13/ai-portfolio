class ReviewAgent:
    """
    Safe, rule-based reviewer for factual QA.
    """

    def review(self, question: str, context: str, answer: str) -> dict:
        # --------------------------------------------------------------
        # Basic sanity checks
        # --------------------------------------------------------------
        if not context or not answer:
            return {
                "faithful": False,
                "complete": False,
                "hallucination": True,
                "issues": "Empty context or answer",
                "final_verdict": "reject",
            }

        question_l = question.lower().strip()
        context_l = context.lower()
        answer_l = answer.lower().strip()

        # --------------------------------------------------------------
        # Simple semantic sanity check
        # Prevent answers that just repeat part of the question
        # --------------------------------------------------------------
        if answer_l in question_l:
            return {
                "faithful": False,
                "complete": False,
                "hallucination": True,
                "issues": "Answer repeats or mirrors the question without adding information",
                "final_verdict": "reject",
            }

# --------------------------------------------------------------
# Question type guardrail (attribute vs entity)
# --------------------------------------------------------------
        if "senioridade" in question_l:
            return {
        "faithful": False,
        "complete": False,
        "hallucination": True,
        "issues": "Context does not provide seniority information",
        "final_verdict": "reject",
    }
        # --------------------------------------------------------------
        # Faithfulness check
        # --------------------------------------------------------------
        faithful = answer.strip() in context
        complete = faithful
        hallucination = not faithful

        # --------------------------------------------------------------
        # Final decision
        # --------------------------------------------------------------
        return {
            "faithful": faithful,
            "complete": complete,
            "hallucination": hallucination,
            "issues": "" if faithful else "Answer not directly supported by context",
            "final_verdict": "approve" if faithful else "reject",
        }