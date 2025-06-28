from abc import ABC, abstractmethod
from typing import List

class BasePreprocessor(ABC):
    """
    Abstract base class for preprocessing raw documents.
    May include cleaning, normalization, or format transformation.
    """

    @abstractmethod
    def preprocess(self, docs: List[str]) -> List[str]:
        """
        Takes a list of raw documents and returns cleaned/preprocessed documents.
        """
        pass