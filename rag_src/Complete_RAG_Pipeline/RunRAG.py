from typing import List, Optional, Dict, Any
import os

from rag_src.llm import BaseLLM, DefaultLLM
from rag_src.retriever import BaseRetriever, DefaultRetriever
from rag_src.embedder import BaseEmbedder, DefaultEmbedder
from rag_src.query_transformer import BaseQueryTransformer, DefaultQueryTransformer
from rag_src.doc_context_enricher import BaseContextEnricher, DefaultContextEnricher
from rag_src.indexer import BaseIndexer, DefaultIndexer
from rag_src.doc_loader import BaseDocLoader, DefaultDocLoader
from rag_src.doc_preprocessor import BasePreprocessor, DefaultPreprocessor
from rag_src.chunker import BaseChunker, DefaultChunker


class RunRAG:
    """
    Full RAG pipeline: indexing + answering
    """

    def __init__(
        self,
        llm: Optional[BaseLLM],
        embeddor: Optional[BaseEmbedder],
        indexer: Optional[BaseIndexer],
        retriever: Optional[BaseRetriever],
        query_transform: Optional[BaseQueryTransformer],
        doc_enricher: Optional[BaseContextEnricher],
        doc_loader: Optional[BaseDocLoader],
        preprocessor: Optional[BasePreprocessor],
        docdir: str,
        chunker: Optional[BaseChunker] = None,
    ):
        self.docdir = docdir
        self.llm = llm or DefaultLLM()
        self.embeddor = embeddor or DefaultEmbedder()
        self.indexer = indexer or DefaultIndexer()
        self.query_transform = query_transform or DefaultQueryTransformer()
        self.doc_enricher = doc_enricher or DefaultContextEnricher()
        self.preprocessor = preprocessor or DefaultPreprocessor()
        self.chunker = chunker or DefaultChunker()

        # Set up doc loader, inject docdir if needed
        if doc_loader:
            if getattr(doc_loader, "path", None) is None:
                doc_loader.path = self.docdir
            self.doc_loader = doc_loader
        else:
            self.doc_loader = DefaultDocLoader(self.docdir)

        # Ensure index exists or build it
        self.ensure_index_ready()

        # Initialize retriever (after index exists)
        index_path = getattr(self.indexer, "persist_path", "default_index")
        self.retriever = retriever or DefaultRetriever(index_path=index_path)

    def ensure_index_ready(self):
        """
        Check if FAISS index exists. If not, run the ingestion pipeline.
        """
        index_path = getattr(self.indexer, "persist_path", "default_index")
        index_file = os.path.join(index_path, "index.faiss")

        if not os.path.exists(index_file):
            print(f"[INFO] FAISS index not found at {index_file}. Running ingestion pipeline.")
            docs = self.load_preprocess_chunk_documents()
            enriched_docs = self.doc_enricher.enrich(docs)
            print(f"Enriched documents count: {len(enriched_docs)}")
            self.ingest_documents(enriched_docs)
        else:
            print(f"[INFO] Found existing index at {index_file}. Skipping ingestion.")

    def load_preprocess_chunk_documents(self) -> List[str]:
        print("=== LOADING DOCUMENTS ===")
        documents = self.doc_loader.load()
        print(f"Loaded {len(documents)} raw documents.")

        if not documents:
            raise RuntimeError("No documents found by doc_loader.")

        if self.preprocessor:
            documents = self.preprocessor.preprocess(documents)
            print(f"Preprocessed down to {len(documents)} documents.")

        if self.chunker:
            documents = self.chunker.chunk(documents)
            print(f"Chunked into {len(documents)} total chunks.")

        return documents

    def ingest_documents(self, documents: List[str], metadata: Optional[List[dict]] = None) -> None:
        print("=== INDEXING DOCUMENTS ===")
        embeddings = self.embeddor.embed(documents)
        self.indexer.index(embeddings, documents, metadata)
        self.indexer.persist()
        print("[INFO] Index persisted.")

    def run(self, query: str) -> str:
        print("=== RUNNING FULL RAG PIPELINE ===")

        # Step 1: Load, Preprocess, Chunk
        docs = self.load_preprocess_chunk_documents()

        # Step 2: Enrich documents
        enriched_docs = self.doc_enricher.enrich(docs)
        print(f"Enriched documents count: {len(enriched_docs)}")

        # Step 3: Embed and Index
        self.ingest_documents(enriched_docs)

        # Step 4: Transform query
        queries = self.query_transform.transform(query)
        print(f"Transformed queries: {queries}")

        # Step 5: Retrieve + Generate
        answers = []
        for i, q in enumerate(queries):
            print(f"Retrieving context for query {i + 1}: {q}")
            context_docs = self.retriever.retrieve(q)
            print(f"Retrieved {len(context_docs)} docs")

            context_text = [doc.get("text", "") for doc in context_docs if doc.get("text", "").strip()]
            final_answer = self.llm.generate(query=q, contexts=context_text)
            answers.append(str(final_answer))

        print("=== PIPELINE COMPLETE ===")
        return "\n\n".join(answers)