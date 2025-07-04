# rag_src/complete_RAG_Pipeline/ReliableRAG.py

from typing import List, Optional
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
        self.retriever = retriever or DefaultRetriever(index_path=getattr(self.indexer, "persist_path", "default_index"))
        self.web_retriever = web_retriever or TavilyWebRetriever()
        self.query_transform = query_transform or DefaultQueryTransformer()
        self.doc_loader = doc_loader or DefaultDocLoader(self.docdir)
        self.preprocessor = preprocessor or DefaultPreprocessor()
        self.chunker = chunker or DefaultChunker()
        self.doc_enricher = doc_enricher or PostDefaultEnricher()

        self.relevance_grader = RelevanceEvaluator(llm=self.llm)
        self.hallucination_grader = DefaultEvaluator(llm=self.llm)
        self.segment_attributor = SegmentAttributor(llm=self.llm)

        index_path = getattr(self.indexer, "persist_path", "default_index")
        index_file = os.path.join(index_path, "index.faiss")
        if not os.path.exists(index_file):
            print("[INFO] FAISS index not found. Running ingestion.")
            self.load_and_ingest_documents()

    def run(self, query: str) -> str:
        print("=== RUNNING RELIABLE RAG PIPELINE ===")
        queries = self.query_transform.transform(query)
        print(f"Step 1: Transformed queries: {queries}")

        retrieved_nodes = []
        seen = set()
        for q in queries:
            results = self.retriever.retrieve(q)
            for node in results:
                if node["text"] not in seen:
                    text_node = TextNode(text=node["text"], metadata=node.get("metadata", {}))
                    retrieved_nodes.append(text_node)
                    seen.add(node["text"])
        print(f"Step 2: Retrieved {len(retrieved_nodes)} internal docs")

        # Evaluate relevance
        context_strings = [n.text for n in retrieved_nodes]
        is_relevant = self.relevance_grader.evaluate(query, response="", contexts=context_strings).get("above_threshold")

        if not is_relevant:
            print("[INFO] Fallback to Web Search")
            retrieved_nodes = self.web_retriever.retrieve(query)

        # Enrich context
        enriched_nodes = self.doc_enricher.enrich(retrieved_nodes)
        context = "\n".join([n.text for n in enriched_nodes])

        # Format answer prompt
        sources = [f"{n.metadata.get('source_url', 'internal')}" for n in enriched_nodes]
        source_text = "\n".join(sources)
        prompt = PromptTemplate(
            "Use the following knowledge to answer the query.\n"
            "Query: {query}\n\nKnowledge:\n{context}\n\nSources:\n{sources}\n\nAnswer:"
        )
        formatted_prompt = prompt.format(query=query, context=context, sources=source_text)
        answer = self.llm.generate(formatted_prompt, contexts=[])
        print(f"Step 3: LLM Answer Generated")

        # Step 4: Hallucination Grading
        hallucination_result = self.hallucination_grader.evaluate(query, response=answer, contexts=[n.text for n in enriched_nodes])
        print(f"Step 4: Hallucination Detected: {hallucination_result.get('hallucination_detected')}")
        print("Verdict:", hallucination_result.get("verdict"))

        # Step 5: Segment Attribution
        attribution = self.segment_attributor.locate_segments(query, answer, enriched_nodes)

        print("\n=== FINAL OUTPUT ===")
        return answer

    def ingest_documents(self, documents: List[str], metadata: Optional[List[dict]] = None) -> None:
        print("[INFO] Indexing documents...")
        embeddings = self.embeddor.embed(documents)
        self.indexer.index(embeddings, documents, metadata)
        self.indexer.persist()
        print("[INFO] Indexing complete.")

    def load_and_ingest_documents(self) -> None:
        print("[INFO] Loading and processing documents...")
        documents = self.doc_loader.load()
        if self.preprocessor:
            documents = self.preprocessor.preprocess(documents)
        if self.chunker:
            documents = self.chunker.chunk(documents)
        self.ingest_documents(documents)
