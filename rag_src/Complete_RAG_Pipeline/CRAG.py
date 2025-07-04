from typing import List, Optional
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

        index_path = getattr(self.indexer, "persist_path", "default_index")
        index_file = os.path.join(index_path, "index.faiss")

        if not os.path.exists(index_file):
            print(f"[INFO] FAISS index not found at {index_file}. Running ingestion pipeline.")
            self.load_and_ingest_documents()
        else:
            print(f"[INFO] Found existing index at {index_file}. Skipping ingestion.")

        self.retriever = retriever or DefaultRetriever(index_path=index_path)

    def run(self, query: str) -> str:
        print("=== RUNNING CRAG PIPELINE (LLAMAINDEX) ===")
        queries = self.query_transform.transform(query)
        print(f"Step 1: Transformed queries: {queries}")

        retrieved_nodes = []
        seen = set()

        for q in queries:
            results = self.retriever.retrieve(q)
            for result in results:
                text = result.get("text", "")
                metadata = result.get("metadata", {})
                if text not in seen:
                    node = TextNode(text=text, metadata=metadata)
                    retrieved_nodes.append(node)
                    seen.add(text)

        print(f"Step 2: Retrieved {len(retrieved_nodes)} internal documents")

        # Evaluate document relevance
        context_strings = [node.text for node in retrieved_nodes]
        eval_result = self.evaluator.evaluate(query, response="", contexts=context_strings)

        if eval_result.get("above_threshold"):
            final_nodes = retrieved_nodes
            sources = [("Local Document", "") for _ in final_nodes]
            print("[INFO] Using internal index")
        else:
            final_nodes = self.web_retriever.retrieve(query)
            sources = [("Web Search", node.metadata.get("source_url", "")) for node in final_nodes]
            print("[INFO] Falling back to web search")

        enriched_nodes = self.doc_enricher.enrich(final_nodes)
        print(f"Step 3: Enriched {len(enriched_nodes)} documents")

        context = "\n".join([node.text for node in enriched_nodes])
        source_text = "\n".join([f"{s[0]}: {s[1]}" if s[1] else s[0] for s in sources])

        prompt = PromptTemplate(
            "Use the following knowledge to answer the query.\n"
            "Query: {query}\n\nKnowledge:\n{context}\n\nSources:\n{sources}\n\nAnswer:"
        )

        formatted_prompt = prompt.format(query=query, context=context, sources=source_text)
        answer = self.llm.generate(formatted_prompt, contexts=[])

        print(f"Step 4: Final Answer: {answer}")
        return answer
