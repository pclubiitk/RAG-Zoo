from rag_src.evaluator.base import BaseEvaluator
from typing import List, Dict, Any
from rag_src.llm import BaseLLM  # Assuming you already have BaseLLM implemented


class RelevanceEvaluator(BaseEvaluator):
    """
    Evaluates how relevant the generated response is to the original query
    using the provided contexts.
    """

    def __init__(self, llm: BaseLLM, threshold: float = 0.7):
        self.llm = llm
        self.threshold = threshold

    def evaluate(
        self,
        query: str,
        response: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        context_str = "\n\n".join(contexts)

        prompt = f"""Evaluate the relevance of the response to the query using the given context.

Query:
{query}

Context:
{context_str}

Response:
{response}

On a scale of 0.0 to 1.0, how relevant is the response to the query based on the context?
Reply with only the score as a float. Do not include explanation."""

        raw_score = self.llm.generate(query=prompt, contexts=[])

        try:
            score = float(raw_score.strip())
        except ValueError:
            score = 0.0  # fallback in case LLM fails

        return {
            "relevance_score": score,
            "above_threshold": score >= self.threshold
        }
