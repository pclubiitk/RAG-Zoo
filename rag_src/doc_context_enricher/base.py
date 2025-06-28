
from abc import ABC, abstractmethod
from typing import List

class BaseContextEnricher(ABC):
    """
    Abstract base class for enriching documents before chunking.
    Could include metadata injection, summaries, QA pairs, etc.
    """

    @abstractmethod
    def enrich(self, docs: List[str]) -> List[str]:
        """
        Enriches the input documents with additional context or structure.

        Args:
            docs (List[str]): Raw or preprocessed documents.

        Returns:
            List[str]: Enriched documents ready for chunking.
        """
        pass
