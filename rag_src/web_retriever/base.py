from abc import ABC, abstractmethod
from typing import List
from llama_index.core.schema import TextNode


class BaseWebRetriever(ABC):
    """
    Abstract base class for WebRetrievers that return relevant online documents.
    """

    @abstractmethod
    def retrieve(self, query: str) -> List[TextNode]:
        """
        Perform a web search and return a list of LlamaIndex TextNodes.
        Each node can include metadata such as source_url.
        """
        pass
