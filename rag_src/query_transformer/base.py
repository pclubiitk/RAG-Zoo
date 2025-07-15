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

    def __call__(self, query: str) -> List[str]:
        """
        Makes the class instance callable like a function.

        Example:
            queries = query_transform("What is YOLO?")

        Returns:
            Same as self.transform(query)
        """
        return self.transform(query)
