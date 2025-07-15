from abc import ABC, abstractmethod
from typing import List, Union


class BaseLLM(ABC):
    """
    Abstract base class for LLM interaction in a RAG system.
    """

    @abstractmethod
    def generate(self, query: str, contexts: List[str]) -> Union[str, dict]:
        """
        Given a user query and a list of contextual documents, return the LLM-generated response.

        Args:
            query (str): The user query.
            contexts (List[str]): Retrieved and relevant document snippets.

        Returns:
            Union[str, dict]: The model-generated response, optionally with metadata.
        """
        pass
