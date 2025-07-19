from typing import List
import re
from .base import BasePreprocessor


class DefaultPreprocessor(BasePreprocessor):
    """
    Default preprocessor that:
    - Lowercases text
    - Strips whitespace
    - Collapses multiple spaces into one
    """

    def preprocess(self, docs: List[str]) -> List[str]:
        cleaned_docs = []
        for doc in docs:
            cleaned = doc.lower()
            cleaned = cleaned.strip()
            cleaned = re.sub(r"\s+", " ", cleaned)  # collapse multiple spaces
            cleaned_docs.append(cleaned)
        return cleaned_docs
