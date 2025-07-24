from typing import List, Optional, Dict, Any
import os

from rag_src.llm import BaseLLM, DefaultLLM
from rag_src.retriever import BaseRetriever, DefaultRetriever
from rag_src.embedder import BaseEmbedder, DefaultEmbedder
from rag_src.query_transformer import BaseQueryTransformer, DefaultQueryTransformer
from rag_src.pre_embedding_enricher import PreBaseEnricher, PreDefaultEnricher
from rag_src.indexer import BaseIndexer, DefaultIndexer
from rag_src.doc_loader import BaseDocLoader, DefaultDocLoader
from rag_src.doc_preprocessor import BasePreprocessor, DefaultPreprocessor
from rag_src.chunker import BaseChunker, DefaultChunker
import asyncio
from typing import AsyncGenerator

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
        doc_enricher: Optional[PreBaseEnricher],
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
        self.doc_enricher = doc_enricher or PreDefaultEnricher()
        self.preprocessor = preprocessor or DefaultPreprocessor()
        self.chunker = chunker or DefaultChunker()

        # Set up doc loader, inject docdir if needed
        if doc_loader:
            if getattr(doc_loader, "path", None) is None:
                doc_loader.path = self.docdir
            self.doc_loader = doc_loader
        else:
            self.doc_loader = DefaultDocLoader(self.docdir)
            
        self.retriever = retriever
        
    async def create_retriever(self):
        await self.ensure_index_ready()
        # Initialize retriever (after index exists)
        index_path = getattr(self.indexer, "persist_path", "default_index")
        self.retriever = self.retriever or DefaultRetriever(index_path=index_path)
        return self
    
    
    async def ensure_index_ready(self):
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
            await self.index_documents(enriched_docs)
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

    async def embed_and_index(self, docs, meta):
        embeddings = await asyncio.to_thread(self.embeddor.embed, docs)
        await asyncio.to_thread(self.indexer.index, embeddings, docs, meta)

    async def index_documents(self, documents: List[str], metadata: Optional[List[dict]] = None, batch_size=16) -> None:
        print("=== INDEXING DOCUMENTS ===")
        batches = [
            (documents[i:i+batch_size], metadata[i:i+batch_size] if metadata else [{}]*batch_size)
            for i in range(0, len(documents), batch_size)
        ]
        tasks = [
            self.embed_and_index(docs, meta)
            for docs, meta in batches
        ]
        await asyncio.gather(*tasks)
        self.indexer.persist()
        print("[INFO] Index persisted.")

    async def _retrieve_and_generate(self, query: str, idx: int) -> str:
        print(f"Retrieving context for query {idx+1} : {query} ")
        context_docs = await self.retriever.retrieve(query)
        print(f"[Query {idx + 1}] Retrieved {len(context_docs)} docs")
        context_texts = [doc.get("text", "") for doc in context_docs if doc.get("text", "").strip()]
        
        if hasattr(self.llm,"generate_stream"):
            return context_texts
        else: 
            print(f"[Query {idx + 1}] Generating response...")
            answer = await asyncio.to_thread(self.llm.generate, query=query, contexts=context_texts)
            return str(answer)
    
    async def run(self, query: str) -> AsyncGenerator[str, None]:
        print("=== RUNNING FULL RAG PIPELINE ===")
        
        # Step 1: Load, Preprocess, Chunk
        await self.create_retriever()
        docs = self.load_preprocess_chunk_documents()

        # Step 2: Enrich documents
        enriched_docs = await self.doc_enricher.enrich(docs)
        print(f"Enriched documents count: {len(enriched_docs)}")

        # Step 3: Embed and Index
        await self.index_documents(enriched_docs)

        # Step 4: Transform query
        queries = self.query_transform.transform(query)
        print(f"Transformed queries: {queries}")

        # Step 5: Retrieve + Generate
        if hasattr(self.llm,"generate_stream"):       
            contexts = await asyncio.gather(*[self._retrieve_and_generate(q, i) for i, q in enumerate(queries)])
            for i, (query, context) in enumerate(zip(queries, contexts)):
                print(f"[Query {i + 1}] Generating response...")
                async for token in self.llm.generate_stream(query, context):
                    yield token
                    yield "\n"
        else:
            answers=await asyncio.gather(*[self._retrieve_and_generate(q,i) for q,i in enumerate(queries)])
            yield "\n\n".join(answers)
        print("=== PIPELINE COMPLETE ===")