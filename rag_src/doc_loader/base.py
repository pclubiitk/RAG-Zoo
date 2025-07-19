from abc import ABC, abstractmethod
from typing import List, Union
from pathlib import Path


class BaseDocLoader(ABC):
    """
    Abstract base class for document loading.
    Responsible for loading raw documents from a source (e.g., filesystem, web).
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)  # Takes a string and covert it to Path object.

    @abstractmethod
    def load(self) -> List[str]:
        """
        Loads raw documents from the path and returns a list of string contents.
        """
        pass
