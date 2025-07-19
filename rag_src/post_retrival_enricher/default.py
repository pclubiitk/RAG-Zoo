from typing import List
from rag_src.post_retrival_enricher.base import PostBaseEnricher


class PostDefaultEnricher(PostBaseEnricher):
    """
    Default context enricher that performs no enrichment.
    Acts as a passthrough for documents.
    """

    def enrich(self, docs: List[str]) -> List[str]:
        return docs
