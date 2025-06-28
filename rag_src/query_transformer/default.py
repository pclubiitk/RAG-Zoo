from .base import BaseQueryTransformer
from typing import List

class DefaultQueryTransformer(BaseQueryTransformer):
    """
    Default transformer that returns the original query as-is.
    Useful when no transformation (HyDE, rephrasing, etc.) is needed.
    """
    def transform(self, query: str) -> List[str]:
        return [query]