# rag_pipeline/base/chunker.py

from abc import ABC, abstractmethod
from typing import List, Optional, Dict


class BaseChunker(ABC):
    """
    Abstract base class for chunking documents into smaller parts.
    Supports optional metadata injection (e.g., titles, summaries).
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(
        self, docs: List[str], metadata: Optional[List[Dict[str, str]]] = None
    ) -> List[str]:
        """
        Splits each document into overlapping or non-overlapping chunks.
        Optionally prepends metadata like titles or summaries to each chunk.

        Args:
            docs (List[str]): Enriched documents.
            metadata (Optional[List[Dict[str, str]]]): Metadata per document (e.g. title, summary).

        Returns:
            List[str]: A flat list of enriched text chunks.
        """
        pass
