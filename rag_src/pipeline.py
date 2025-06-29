from typing import List, Optional
from rag_src.query_transformer import BaseQueryTransform
from rag_src.doc_context_enricher import BaseDocEnricher
from rag_src.retriever import BaseRetriever


class RunRAG:
    """
    Run the full RAG pipeline: query transformation → retrieval → document enrichment.
    """

    def __init__(
        self,
        llm: object,
        embeddor: object,
        vectorstore: object,
        docdir: str,
        query_transform: Optional[BaseQueryTransform],
        doc_enricher: Optional[BaseDocEnricher],
        retriever: Optional[BaseRetriever],
    ):
        self.llm = llm
        self.embeddor = embeddor
        self.vectorstore = vectorstore
        self.docdir = docdir
        self.query_transform = query_transform
        self.doc_enricher = doc_enricher
        self.retriever = retriever

    def run(self, query: str) -> List[str]:
        print("=== RUNNING RAG PIPELINE ===")

        # Step 1: Transform query
        queries = self.query_transform(query) if self.query_transform else [query]
        print(f"Step 1: Transformed queries: {queries}")

        # Step 2: Retrieve documents
        docs = self.retriever.retrieve(queries)
        print(f"Step 2: Retrieved {len(docs)} documents")

        # Step 3: Enrich documents
        enriched_docs = self.doc_enricher.enrich(docs)
        print(f"Step 3: Enriched documents count: {len(enriched_docs)}")

        return enriched_docs
