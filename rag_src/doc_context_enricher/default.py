from typing import List
from rag_src.doc_context_enricher.base import BaseContextEnricher

class DefaultContextEnricher(BaseContextEnricher):
    """
    Default context enricher that performs no enrichment.
    Acts as a passthrough for documents.
    """
    def enrich(self, docs: List[str]) -> List[str]:
        return docs
