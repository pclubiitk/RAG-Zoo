from abc import ABC, abstractmethod
from typing import List, Literal


class BaseEmbedder(ABC):
    """
    Abstract base class for embedding documents or queries into vector representations.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

    @abstractmethod
    def embed(
        self, texts: List[str], mode: Literal["query", "document"] = "document"
    ) -> List[List[float]]:
        """
        Converts input texts (chunks or queries) into embeddings.

        Args:
            texts (List[str]): List of strings to embed.
            mode (Literal["query", "document"]): Embedding mode. Some models treat query/document differently.

        Returns:
            List[List[float]]: List of dense vector embeddings.
        """
        pass
