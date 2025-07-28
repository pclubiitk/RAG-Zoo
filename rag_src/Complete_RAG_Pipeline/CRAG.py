from typing import List, Optional, AsyncGenerator
import os

from rag_src.llm import BaseLLM, DefaultLLM
from rag_src.retriever import BaseRetriever, DefaultRetriever
from rag_src.web_retriever import BaseWebRetriever
from rag_src.web_retriever import TavilyWebRetriever
from rag_src.embedder import BaseEmbedder, DefaultEmbedder
from rag_src.query_transformer import BaseQueryTransformer, LLMWebQueryTransformer
from rag_src.post_retrival_enricher import PostBaseEnricher, PostDefaultEnricher
from rag_src.indexer import BaseIndexer, DefaultIndexer
from rag_src.doc_loader import BaseDocLoader, DefaultDocLoader
from rag_src.doc_preprocessor import BasePreprocessor, DefaultPreprocessor
from rag_src.chunker import BaseChunker, DefaultChunker
from rag_src.evaluator.base import BaseEvaluator
from rag_src.evaluator import RelevanceEvaluator

from llama_index.core.schema import TextNode
from llama_index.core.prompts import PromptTemplate
import asyncio


class CRAG:
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        embeddor: Optional[BaseEmbedder] = None,
        indexer: Optional[BaseIndexer] = None,
        retriever: Optional[BaseRetriever] = None,
        web_retriever: Optional[BaseWebRetriever] = None,
        evaluator: Optional[BaseEvaluator] = None,
        query_transform: Optional[BaseQueryTransformer] = None,
        doc_enricher: Optional[PostBaseEnricher] = None,
        doc_loader: Optional[BaseDocLoader] = None,
        preprocessor: Optional[BasePreprocessor] = None,
        chunker: Optional[BaseChunker] = None,
        docdir: str = "data",
    ):
        self.docdir = docdir
        self.llm = llm or DefaultLLM()
        self.embeddor = embeddor or DefaultEmbedder()
        self.indexer = indexer or DefaultIndexer()
        self.query_transform = query_transform or LLMWebQueryTransformer(self.llm)
        self.doc_enricher = doc_enricher or PostDefaultEnricher()
        self.doc_loader = doc_loader or DefaultDocLoader(self.docdir)
        self.preprocessor = preprocessor or DefaultPreprocessor()
        self.chunker = chunker or DefaultChunker()

        self.evaluator = evaluator or RelevanceEvaluator(llm=self.llm)
        self.web_retriever = web_retriever or TavilyWebRetriever()
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
            docs = self.load_preprocess_chunk_documents()
            enriched_docs = await self.doc_enricher.enrich(docs)
            await self.index_documents(enriched_docs)

        else:
            print(f"[INFO] Found existing index at {index_file}. Skipping ingestion.")
            
    def load_preprocess_chunk_documents(self) -> List[str]:
        documents = self.doc_loader.load()
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

    async def run(self, query: str) -> AsyncGenerator[str, None]:
        print("=== RUNNING CRAG PIPELINE (LLAMAINDEX) ===")
        docs = self.load_preprocess_chunk_documents()
        await self.create_retriever()
        queries = self.query_transform.transform(query)
        print(f"Step 1: Transformed queries: {queries}")
        retrieved_nodes = []
        seen = set()
        results = await asyncio.gather(*(self.retriever.retrieve(q) for q in queries))
        for sublist in results:
            for result in sublist:
                text = result.get("text", "")
                metadata = result.get("metadata", {})
                if text not in seen:
                    node = TextNode(text=text, metadata=metadata)
                    retrieved_nodes.append(node)
                    seen.add(text)
        print(f"Step 2: Retrieved {len(retrieved_nodes)} internal documents")
        context_strings = [node.text for node in retrieved_nodes]
        eval_result = await self.evaluator.evaluate(query, response="", contexts=context_strings)
        if eval_result.get("above_threshold"):
            final_nodes = retrieved_nodes
            sources = [("Local Document", "") for _ in final_nodes]
            print("[INFO] Using internal index")
        else:
            final_nodes = await self.web_retriever.retrieve(query)
            sources = [("Web Search", node.metadata.get("source_url", "")) for node in final_nodes]
            print("[INFO] Falling back to web search")
        enriched_nodes = await self.doc_enricher.enrich(final_nodes)
        print(f"Step 4: Enriched {len(enriched_nodes)} documents")

        context = "\n".join([node.text for node in enriched_nodes])
        source_text = "\n".join([f"{s[0]}: {s[1]}" if s[1] else s[0] for s in sources])

        prompt = PromptTemplate(
            "Use the following knowledge to answer the query.\n"
            "Query: {query}\n\nKnowledge:\n{context}\n\nSources:\n{sources}\n\nAnswer:"
        )
        formatted_prompt = prompt.format(query=query, context=context, sources=source_text)
        print(f"Step 5: Prompt ready")
        print(f"Step 6: Final Answer:")
        if hasattr(self.llm, "generate_stream"):
            async for token in self.llm.generate_stream(formatted_prompt, contexts=[]):
                yield token
            yield "\n"
        else:
            answer = await self.llm.generate(formatted_prompt, contexts=[])
            yield "\n\n".join(answer)