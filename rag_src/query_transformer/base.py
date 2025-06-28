from abc import ABC, abstractmethod
from typing import List

class BaseQueryTransformer(ABC):
    """
    Abstract base class for transforming a user query.
    Used for multi-query expansion, hallucination (HyDE), or rephrasing.
    """

    @abstractmethod
    def transform(self, query: str) -> List[str]:
        """
        Takes in a raw query and returns a list of transformed queries.

        Args:
            query (str): The original user query.

        Returns:
            List[str]: One or more reformulated or hallucinated queries.
        """
        pass
