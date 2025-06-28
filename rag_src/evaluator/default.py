from typing import List, Dict, Any, Union
from .base import BaseEvaluator

class DefaultEvaluator(BaseEvaluator):
    """
    LLM-based evaluator that checks for hallucinations in the response.
    It asks the LLM: Is the answer fully supported by the given context?
    """
    def __init__(self, llm):
        self.llm = llm  # Must follow BaseLLM interface

    def evaluate(
        self,
        query: str,
        response: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        check_prompt = (
            "You are an evaluator checking for hallucinations in a generated answer.\n\n"
            f"Context:\n{''.join(contexts)}\n\n"
            f"Question:\n{query}\n\n"
            f"Answer:\n{response}\n\n"
            "Is the answer fully supported by the context above? "
            "Reply with 'Yes' or 'No', followed by a one-line justification."
        )

        verdict = self.llm.generate(check_prompt, contexts=[])  # Pass empty contexts if not used

        result_text = verdict["text"] if isinstance(verdict, dict) else verdict
        result_text = result_text.strip()

        hallucinated = "no" in result_text.lower().splitlines()[0]

        return {
            "hallucination_detected": hallucinated,
            "verdict": result_text
        }