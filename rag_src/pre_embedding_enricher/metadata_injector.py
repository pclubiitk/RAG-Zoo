

from rag_src.pre_embedding_enricher.base  import PreBaseEnricher
from typing import List

class MetadataInjector(PreBaseEnricher):
    """
    Adds metadata (like title, author, timestamp) to each document.
    Metadata is provided as a dictionary indexed by document position.
    """

    def __init__(self, metadata: dict):
        self.metadata = metadata

    def enrich(self, docs: List[str]) -> List[str]:
        enriched_docs = []
        for i, doc in enumerate(docs):
            meta = self.metadata.get(i, "")
            enriched_doc = f"{meta}\n\n{doc}" if meta else doc
            enriched_docs.append(enriched_doc)
        return enriched_docs