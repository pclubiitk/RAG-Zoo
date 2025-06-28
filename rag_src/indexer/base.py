# rag_pipeline/base/indexer.py

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class BaseIndexer(ABC):
    """
    Abstract base class for indexing embedded documents into a vector store.
    """

    @abstractmethod
    def index(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Indexes a list of embeddings and corresponding documents into the vector store.

        Args:
            embeddings (List[List[float]]): Vector representations of documents.
            documents (List[str]): Original document chunks.
            metadata (Optional[List[Dict[str, Any]]]): Optional metadata for each document.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the index (clears all stored vectors and metadata).
        """
        pass

    @abstractmethod
    def persist(self) -> None:
        """
        Persists the current index to disk or cloud (if supported).
        """
        pass
