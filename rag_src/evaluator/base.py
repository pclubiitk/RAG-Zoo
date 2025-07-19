

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseEvaluator(ABC):
    """
    Abstract base class for evaluating RAG outputs such as responses or retrieved context.
    """

    @abstractmethod
    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of the response given the query and supporting context.

        Args:
            query (str): The original user query.
            response (str): The LLM-generated answer.
            contexts (List[str]): Retrieved context documents used to generate the response.

        Returns:
            Dict[str, Any]: Evaluation scores or metrics (e.g., relevance, consistency, score).
        """
        pass
