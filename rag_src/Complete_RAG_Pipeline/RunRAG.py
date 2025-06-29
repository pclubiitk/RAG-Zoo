from typing import List, Optional
from rag_src.llm import BaseLLM, DefaultLLM
from rag_src.retriever import BaseRetriever, DefaultRetriever
from rag_src.embedder import BaseEmbedder, DefaultEmbedder
from rag_src.query_transformer import BaseQueryTransformer, DefaultQueryTransformer
from rag_src.doc_context_enricher import BaseContextEnricher, DefaultContextEnricher
from rag_src.indexer import BaseIndexer, DefaultIndexer
from rag_src.doc_loader import BaseDocLoader, DefaultDocLoader
from rag_src.doc_preprocessor import BasePreprocessor, DefaultPreprocessor

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
    ):
        self.llm = llm or DefaultLLM()
        self.embeddor = embeddor or DefaultEmbedder()
        self.indexer = indexer or DefaultIndexer()
        self.retriever = retriever or DefaultRetriever()
        self.query_transform = query_transform or DefaultQueryTransformer()
        self.doc_enricher = doc_enricher or DefaultContextEnricher()
        self.doc_loader = doc_loader or DefaultDocLoader(self.docdir)
        self.preprocessor = preprocessor or DefaultPreprocessor()
        self.docdir = docdir

    def run(self, query: str) -> str:
        print("=== RUNNING RAG PIPELINE ===")

        queries = self.query_transform(query) if self.query_transform else [query]
        print(f"Step 1: Transformed queries: {queries}")

        docs = self.retriever.retrieve(queries)
        print(f"Step 2: Retrieved {len(docs)} documents")

        enriched_docs = self.doc_enricher.enrich(docs)
        print(f"Step 3: Enriched documents count: {len(enriched_docs)}")

        final_answer = self.llm.generate(query, enriched_docs)
        print(f"Step 4: Final Answer: {final_answer}")

        return final_answer

    def ingest_documents(self, documents: List[str], metadata: Optional[List[dict]] = None) -> None:
        """
        Manually index given documents.
        """
        if not self.embeddor or not self.indexer:
            raise ValueError("Embedder or indexer not set.")

        print("=== INDEXING DOCUMENTS ===")
        embeddings = self.embeddor.embed_documents(documents)
        self.indexer.index(embeddings, documents, metadata)
        self.indexer.persist()
        print("Index persisted.")

    def load_and_ingest_documents(self) -> None:
        """
        Loads documents using the loader, optionally preprocesses them, then indexes.
        """
        if not self.doc_loader:
            raise ValueError("No document loader provided.")
        
        print("=== LOADING DOCUMENTS ===")
        documents = self.doc_loader.load()
        print(f"Loaded {len(documents)} raw documents.")

        if self.preprocessor:
            documents = self.preprocessor.preprocess(documents)
            print(f"Preprocessed down to {len(documents)} documents.")

        self.ingest_documents(documents)




# RUNRAG Demo script(only for testing it.)
'''
from rag_src.rag_pipeline import RunRAG  # or wherever RunRAG is defined

# Initialize with just docdir — everything else uses default classes
rag = RunRAG(
    llm=None,
    embeddor=None,
    indexer=None,
    retriever=None,
    query_transform=None,
    doc_enricher=None,
    doc_loader=None,
    preprocessor=None,
    docdir="docs/"  # directory where sample.txt is saved
)

# Step 1: Ingest the documents
rag.load_and_ingest_documents()

# Step 2: Ask a question
query = "What is the sun made of?"
answer = rag.run(query)

print("\n💡 Answer:")
print(answer)
'''