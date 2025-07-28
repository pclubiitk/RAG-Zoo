from typing import List, Optional, Dict, Any
import os
import time
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
        llm: Optional[BaseLLM] = None,
        embeddor: Optional[BaseEmbedder] = None,
        indexer: Optional[BaseIndexer] = None,
        retriever: Optional[BaseRetriever] = None,
        query_transform: Optional[BaseQueryTransformer] = None,
        doc_enricher: Optional[PreBaseEnricher] = None,
        doc_loader: Optional[BaseDocLoader] = None,
        preprocessor: Optional[BasePreprocessor] = None,
        docdir: str = "data",
        chunker: Optional[BaseChunker] = None,
    ):
        self.docdir = docdir
        self.llm = llm or DefaultLLM()
        self.embeddor = embeddor or DefaultEmbedder()
        self.indexer = indexer or DefaultIndexer()
        self.query_transform = query_transform(self.llm.llm) or DefaultQueryTransformer()
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
        index_ready_start = time.perf_counter()
        await self.ensure_index_ready()
        index_ready_time = time.perf_counter() - index_ready_start
        print(f"[TIMER] ensure_index_ready() took: {index_ready_time:.2f} seconds")

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

            load_start = time.perf_counter()
            docs = self.load_preprocess_chunk_documents()
            load_time = time.perf_counter() - load_start
            print(f"[TIMER] Loading & preprocessing took: {load_time:.2f} seconds")

            enrich_start = time.perf_counter()
            enriched_docs = await self.doc_enricher.enrich(docs)
            enrich_time = time.perf_counter() - enrich_start
            print(f"[TIMER] Enriching documents took: {enrich_time:.2f} seconds")
            print(f"[INFO] Enriched documents count: {len(enriched_docs)}")

            index_start = time.perf_counter()
            await self.index_documents(enriched_docs)
            index_time = time.perf_counter() - index_start
            print(f"[TIMER] Indexing documents took: {index_time:.2f} seconds")

        else:
            print(f"[INFO] Found existing index at {index_file}. Skipping ingestion.")


    def load_preprocess_chunk_documents(self) -> List[str]:
        print("=== LOADING DOCUMENTS ===")

        t_load_start = time.perf_counter()
        documents = self.doc_loader.load()
        t_load_end = time.perf_counter()
        print(f"[TIMER] Loading took {t_load_end - t_load_start:.2f} seconds")
        print(f"Loaded {len(documents)} raw documents.")

        if not documents:
            raise RuntimeError("No documents found by doc_loader.")

        if self.preprocessor:
            t_pre_start = time.perf_counter()
            documents = self.preprocessor.preprocess(documents)
            t_pre_end = time.perf_counter()
            print(f"[TIMER] Preprocessing took {t_pre_end - t_pre_start:.2f} seconds")
            print(f"Preprocessed down to {len(documents)} documents.")

        if self.chunker:
            t_chunk_start = time.perf_counter()
            documents = self.chunker.chunk(documents)
            t_chunk_end = time.perf_counter()
            print(f"[TIMER] Chunking took {t_chunk_end - t_chunk_start:.2f} seconds")
            print(f"Chunked into {len(documents)} total chunks.")

        return documents


    async def embed_and_index(self, docs, meta):
        t_embed_start = time.perf_counter()
        embeddings = await asyncio.to_thread(self.embeddor.embed, docs)
        t_embed_end = time.perf_counter()
        print(f"[TIMER] Embedding took {t_embed_end - t_embed_start:.2f} seconds")

        t_index_start = time.perf_counter()
        await asyncio.to_thread(self.indexer.index, embeddings, docs, meta)
        t_index_end = time.perf_counter()
        print(f"[TIMER] Indexing took {t_index_end - t_index_start:.2f} seconds")


    async def index_documents(self, documents: List[str], metadata: Optional[List[dict]] = None, batch_size=16) -> None:
        print("=== INDEXING DOCUMENTS ===")
        t_total_start = time.perf_counter()

        batches = [
            (documents[i:i + batch_size], metadata[i:i + batch_size] if metadata else [{}] * batch_size)
            for i in range(0, len(documents), batch_size)
        ]

        tasks = [
            self.embed_and_index(docs, meta)
            for docs, meta in batches
        ]

        await asyncio.gather(*tasks)

        self.indexer.persist()
        print("[INFO] Index persisted.")

        t_total_end = time.perf_counter()
        print(f"[TIMER] Total indexing (embed + index) took {t_total_end - t_total_start:.2f} seconds")

    async def _retrieve_and_generate(self, query: str, idx: int) -> str:
        print(f"Retrieving context for query {idx+1} : {query}")
        t0 = time.perf_counter()

        context_docs = await self.retriever.retrieve(query)

        t1 = time.perf_counter()
        print(f"[Query {idx + 1}] Retrieved {len(context_docs)} docs in {t1 - t0:.2f}s")

        context_texts = [doc.get("text", "") for doc in context_docs if doc.get("text", "").strip()]

        if hasattr(self.llm, "generate_stream"):
            return context_texts
        else:
            print(f"[Query {idx + 1}] Generating response...")
            t2 = time.perf_counter()
            answer = await asyncio.to_thread(self.llm.generate, query=query, contexts=context_texts)
            t3 = time.perf_counter()
            print(f"[Query {idx + 1}] Generation took {t3 - t2:.2f}s")
            return str(answer)


    async def run(self, query: str) -> AsyncGenerator[str, None]:
        print("=== RUNNING FULL RAG PIPELINE ===")

        # Step 1: Load, Preprocess, Chunk
        await self.create_retriever()
        t0 = time.perf_counter()
        docs = self.load_preprocess_chunk_documents()
        print(f"[Step 1] Loaded & chunked {len(docs)} documents in {time.perf_counter() - t0:.2f}s")

        # Step 2: Enrich documents
        t1 = time.perf_counter()
        enriched_docs = await self.doc_enricher.enrich(docs)
        print(f"[Step 2] Enriched documents count: {len(enriched_docs)} in {time.perf_counter() - t1:.2f}s")

        # Step 3: Embed and Index
        t2 = time.perf_counter()
        await self.index_documents(enriched_docs)
        print(f"[Step 3] Embedded & indexed in {time.perf_counter() - t2:.2f}s")

        # Step 4: Transform query
        t3 = time.perf_counter()
        queries = self.query_transform.transform(query=query)
        print(f"[Step 4] Transformed queries: {queries} in {time.perf_counter() - t3:.2f}s")

        # Step 5: Retrieve + Generate
        print(f"[Step 5] Processing {len(queries)} queries...")

        if hasattr(self.llm, "generate_stream"):
            # Streamed token-by-token generation
            contexts = await asyncio.gather(*[
                self._retrieve_and_generate(q, i) for i, q in enumerate(queries)
            ])
            for i, (q, context) in enumerate(zip(queries, contexts)):
                print(f"[Query {i + 1}] Streaming response...")
                async for token in self.llm.generate_stream(q, context):
                    yield token
                yield "\n"
        else:
            # Normal generation
            answers = await asyncio.gather(*[
                self._retrieve_and_generate(q, i) for i, q in enumerate(queries)
            ])
            yield "\n\n".join(answers)

        print("=== PIPELINE COMPLETE ===")
