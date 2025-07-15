from typing import List
from rag_src.pre_embedding_enricher.base import PreBaseEnricher


class PreDefaultEnricher(PreBaseEnricher):
    """
    Default context enricher that performs no enrichment.
    Acts as a passthrough for documents.
    """

    def enrich(self, docs: List[str]) -> List[str]:
        return docs
