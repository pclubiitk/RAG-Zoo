from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseRetriever(ABC):
    """
    Abstract base class for retrieving relevant chunks from a vector index based on a query.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    @abstractmethod
    async def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant documents for the given query.

        Args:
            query (str): The user query string.

        Returns:
            List[Dict[str, Any]]: A list of results, each with 'text' and optional metadata fields.
        """
        pass
