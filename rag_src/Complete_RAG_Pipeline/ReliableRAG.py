# rag_src/complete_RAG_Pipeline/ReliableRAG.py

from typing import List, Optional, AsyncGenerator
import os

from rag_src.llm import BaseLLM, DefaultLLM
from rag_src.retriever import BaseRetriever, DefaultRetriever
from rag_src.web_retriever import BaseWebRetriever, TavilyWebRetriever
from rag_src.embedder import BaseEmbedder, DefaultEmbedder
from rag_src.query_transformer import BaseQueryTransformer, DefaultQueryTransformer
from rag_src.post_retrival_enricher import PostBaseEnricher, PostDefaultEnricher
from rag_src.indexer import BaseIndexer, DefaultIndexer
from rag_src.doc_loader import BaseDocLoader, DefaultDocLoader
from rag_src.doc_preprocessor import BasePreprocessor, DefaultPreprocessor
from rag_src.chunker import BaseChunker, DefaultChunker

from llama_index.core.schema import TextNode
from llama_index.core.prompts import PromptTemplate

from rag_src.evaluator.doc_relevance_evaluator import RelevanceEvaluator
from rag_src.evaluator import DefaultEvaluator
from rag_src.evaluator.segment_attributor import SegmentAttributor
import asyncio


class ReliableRAG:
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        embeddor: Optional[BaseEmbedder] = None,
        indexer: Optional[BaseIndexer] = None,
        retriever: Optional[BaseRetriever] = None,
        web_retriever: Optional[BaseWebRetriever] = None,
        query_transform: Optional[BaseQueryTransformer] = None,
        doc_loader: Optional[BaseDocLoader] = None,
        preprocessor: Optional[BasePreprocessor] = None,
        chunker: Optional[BaseChunker] = None,
        doc_enricher: Optional[PostBaseEnricher] = None,
        docdir: str = "data",
    ):
        self.docdir = docdir
        self.llm = llm or DefaultLLM()
        self.embeddor = embeddor or DefaultEmbedder()
        self.indexer = indexer or DefaultIndexer()
        self.web_retriever = web_retriever or TavilyWebRetriever()
        self.query_transform = query_transform or DefaultQueryTransformer()
        self.doc_loader = doc_loader or DefaultDocLoader(self.docdir)
        self.preprocessor = preprocessor or DefaultPreprocessor()
        self.chunker = chunker or DefaultChunker()
        self.doc_enricher = doc_enricher or PostDefaultEnricher()
        self.relevance_grader = RelevanceEvaluator(llm=self.llm)
        self.hallucination_grader = DefaultEvaluator(llm=self.llm)
        self.segment_attributor = SegmentAttributor(llm=self.llm)
        
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

    async def run(self, query: str) ->  AsyncGenerator[str,None]:
        print("=== RUNNING RELIABLE RAG PIPELINE ===")
        await self.create_retriever()
        docs = self.load_preprocess_chunk_documents()
        await self.index_documents(docs)
        queries = self.query_transform.transform(query)
        print(f"Step 1: Transformed queries: {queries}")

        retrieved_nodes = []
        seen = set()
        results = await asyncio.gather(*[self.retriever.retrieve(q) for q in queries])
        for sublist in results:
            for result in sublist:
                text = result.get("text", "")
                metadata = result.get("metadata", {})
                if text not in seen:
                    node = TextNode(text=text, metadata=metadata)
                    retrieved_nodes.append(node)
                    seen.add(text)
        print(f"Step 2: Retrieved {len(retrieved_nodes)} internal docs")

        # Evaluate relevance
        context_strings = [n.text for n in retrieved_nodes]
        is_relevant = await self.relevance_grader.evaluate(query, response="", contexts=context_strings)
        is_relevant = is_relevant.get("above_threshold")

        if not is_relevant:
            print("[INFO] Fallback to Web Search")
            retrieved_nodes = await self.web_retriever.retrieve(query)

        # Enrich context
        enriched_nodes = await self.doc_enricher.enrich(retrieved_nodes)
        context = "\n".join([n.text for n in enriched_nodes])

        # Format answer prompt
        sources = [f"{n.metadata.get('source_url', 'internal')}" for n in enriched_nodes]
        source_text = "\n".join(sources)
        prompt = PromptTemplate(
            "Use the following knowledge to answer the query.\n"
            "Query: {query}\n\nKnowledge:\n{context}\n\nSources:\n{sources}\n\nAnswer:"
        )
        formatted_prompt = prompt.format(query=query, context=context, sources=source_text)
        answer = await self.llm.generate(formatted_prompt, contexts=[])
        print(f"Step 3: LLM Answer Generated")

        # Step 4: Hallucination Grading
        hallucination_result = await self.hallucination_grader.evaluate(query, response=answer, contexts=[n.text for n in enriched_nodes])
        print(f"Step 4: Hallucination Detected: {hallucination_result.get('hallucination_detected')}")
        print("Verdict:", hallucination_result.get("verdict"))

        # Step 5: Segment Attribution
        attribution = await self.segment_attributor.locate_segments(query, answer, enriched_nodes)
        yield answer
        print("\n=== FINAL OUTPUT ===")
        
