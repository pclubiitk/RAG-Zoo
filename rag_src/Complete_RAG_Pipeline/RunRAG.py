from typing import List, Optional
from rag_src.llm import BaseLLM, DefaultLLM
from rag_src.retriever import BaseRetriever, DefaultRetriever
from rag_src.embedder import BaseEmbedder, DefaultEmbedder
from rag_src.query_transformer import BaseQueryTransformer, DefaultQueryTransformer
from rag_src.doc_context_enricher import BaseContextEnricher, DefaultContextEnricher
from rag_src.indexer import BaseIndexer, DefaultIndexer
from rag_src.doc_loader import BaseDocLoader, DefaultDocLoader
from rag_src.doc_preprocessor import BasePreprocessor, DefaultPreprocessor
from rag_src.chunker import BaseChunker, DefaultChunker
import os

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
        self.doc_loader = doc_loader or DefaultDocLoader(self.docdir)
        self.preprocessor = preprocessor or DefaultPreprocessor()
        self.chunker = chunker or DefaultChunker() 

        # Ensure index is built before initializing retriever
        index_path = getattr(self.indexer, "persist_path", "default_index")
        index_file = os.path.join(index_path, "index.faiss")

        if not os.path.exists(index_file):
            print(f"[INFO] FAISS index not found at {index_file}. Running ingestion pipeline.")
            self.load_and_ingest_documents()
        else:
            print(f"[INFO] Found existing index at {index_file}. Skipping ingestion.")

        self.retriever = retriever or DefaultRetriever(index_path=index_path)

        

    def run(self, query: str) -> str:
        print("=== RUNNING RAG PIPELINE ===")

        # Step 1: Transform query
        queries = self.query_transform.transform(query) if self.query_transform else [query]
        print(f"Step 1: Transformed queries: {queries}")

        # Step 2: Retrieve documents for all queries
        all_docs = []
        seen_texts = set()

        for q in queries:
            results = self.retriever.retrieve(q)
            for doc in results:
                if doc["text"] not in seen_texts:
                    all_docs.append(doc)
                    seen_texts.add(doc["text"])

        print(f"Step 2: Retrieved {len(all_docs)} unique documents")

        # Step 3: Enrich
        enriched_docs = self.doc_enricher.enrich(all_docs)
        print(f"Step 3: Enriched documents count: {len(enriched_docs)}")

        # Step 4: LLM Generation
        context_texts = [doc["text"] for doc in enriched_docs]
        final_answer = self.llm.generate(query, context_texts)

        print(f"Step 4: Final Answer: {final_answer}")

        return final_answer


    def ingest_documents(self, documents: List[str], metadata: Optional[List[dict]] = None) -> None:
        """
        Manually index given documents.
        """
        if not self.embeddor or not self.indexer:
            raise ValueError("Embedder or indexer not set.")

        print("=== INDEXING DOCUMENTS ===")
        embeddings = self.embeddor.embed(documents)
        self.indexer.index(embeddings, documents, metadata)
        self.indexer.persist()
        print("Index persisted.")

    def load_and_ingest_documents(self) -> None:
        if not self.doc_loader:
            raise ValueError("No document loader provided.")

        print("=== LOADING DOCUMENTS ===")
        documents = self.doc_loader.load()
        print(f"Loaded {len(documents)} raw documents.")

        if self.preprocessor:
            documents = self.preprocessor.preprocess(documents)
            print(f"Preprocessed down to {len(documents)} documents.")

        if self.chunker:
            documents = self.chunker.chunk(documents)
            print(f"Chunked into {len(documents)} total chunks.")

        self.ingest_documents(documents)


