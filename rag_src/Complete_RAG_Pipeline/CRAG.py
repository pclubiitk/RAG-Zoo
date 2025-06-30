from typing import List, Optional
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

from llama_index.core.schema import TextNode
from llama_index.core.prompts import PromptTemplate


class CRAG:
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        embeddor: Optional[BaseEmbedder] = None,
        indexer: Optional[BaseIndexer] = None,
        retriever: Optional[BaseRetriever] = None,
        query_transform: Optional[BaseQueryTransformer] = None,
        doc_enricher: Optional[BaseContextEnricher] = None,
        doc_loader: Optional[BaseDocLoader] = None,
        preprocessor: Optional[BasePreprocessor] = None,
        chunker: Optional[BaseChunker] = None,
        docdir: str = "data",
    ):
        self.docdir = docdir
        self.llm = llm or DefaultLLM()
        self.embeddor = embeddor or DefaultEmbedder()
        self.indexer = indexer or DefaultIndexer()
        self.retriever = retriever  # initialized below after index is ready
        self.query_transform = query_transform or DefaultQueryTransformer()
        self.doc_enricher = doc_enricher or DefaultContextEnricher()
        self.doc_loader = doc_loader or DefaultDocLoader(self.docdir)
        self.preprocessor = preprocessor or DefaultPreprocessor()
        self.chunker = chunker or DefaultChunker()

        # Ensure index exists
        index_path = getattr(self.indexer, "persist_path", "default_index")
        index_file = os.path.join(index_path, "index.faiss")

        if not os.path.exists(index_file):
            print(f"[INFO] FAISS index not found at {index_file}. Running ingestion pipeline.")
            self.load_and_ingest_documents()
        else:
            print(f"[INFO] Found existing index at {index_file}. Skipping ingestion.")

        self.retriever = retriever or DefaultRetriever(index_path=index_path)

    def _is_relevant(self, query: str, docs: List[TextNode]) -> bool:
        if not docs:
            return False

        context = "\n\n".join([doc.text for doc in docs[:5]])
        prompt = (
            f"You are a helpful assistant. Determine if the following documents contain enough relevant information "
            f"to answer the query.\n\nQuery:\n{query}\n\nDocuments:\n{context}\n\n"
            f"Respond with 'yes' or 'no' only."
        )
        response = self.llm.generate(prompt, contexts=[]).strip().lower()
        return response.startswith("y")

    def _web_retrieve(self, query: str) -> List[TextNode]:
        print(f"[INFO] Performing Tavily web search for: {query}")
    
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY not set in environment.")

        client = TavilyClient(api_key=tavily_api_key)

        try:
            results = client.search(query=query, max_results=5)
            web_nodes = []

            for res in results.get("results", []):
                text = f"{res.get('title', '')}\n{res.get('content', '')}".strip()
                url = res.get("url", "")
                if text:
                    web_nodes.append(TextNode(text=text, metadata={"source_url": url}))
        
            print(f"[INFO] Retrieved {len(web_nodes)} web documents")
            return web_nodes

        except Exception as e:
            print(f"[ERROR] Tavily search failed: {e}")
            return [TextNode(text=f"Web search failed: {str(e)}")]


    def run(self, query: str) -> str:
        print("=== RUNNING CRAG PIPELINE (LLAMAINDEX) ===")
        queries = self.query_transform.transform(query) if self.query_transform else [query]
        print(f"Step 1: Transformed queries: {queries}")

        # Retrieve from internal index
        retrieved_nodes = []
        seen = set()

        for q in queries:
            results = self.retriever.retrieve(q)
            for node in results:
                if node.text not in seen:
                    retrieved_nodes.append(node)
                    seen.add(node.text)

        print(f"Step 2: Retrieved {len(retrieved_nodes)} internal documents")

        # Relevance Check
        if self._is_relevant(query, retrieved_nodes):
            final_nodes = retrieved_nodes
            sources = [("Local Document", "")]
            print("[INFO] Using internal index")
        else:
            web_nodes = self._web_retrieve(query)
            final_nodes = web_nodes
            sources = [("Web Search", "") for _ in web_nodes]
            print("[INFO] Falling back to web search")

        # Enrich
        enriched_nodes = self.doc_enricher.enrich(final_nodes)
        print(f"Step 3: Enriched {len(enriched_nodes)} documents")

        # LLM Generation
        context = "\n".join([node.text for node in enriched_nodes])
        source_text = "\n".join([f"{s[0]}: {s[1]}" if s[1] else s[0] for s in sources])

        template = PromptTemplate(
            "Use the following knowledge to answer the query.\n"
            "Query: {query}\n\nKnowledge:\n{context}\n\nSources:\n{sources}\n\nAnswer:"
        )

        full_prompt = template.format(query=query, context=context, sources=source_text)
        final_answer = self.llm.generate(full_prompt, contexts=[])
        print(f"Step 4: Final Answer: {final_answer}")
        return final_answer

    def ingest_documents(self, documents: List[str], metadata: Optional[List[dict]] = None) -> None:
        print("=== INDEXING DOCUMENTS ===")
        embeddings = self.embeddor.embed(documents)
        self.indexer.index(embeddings, documents, metadata)
        self.indexer.persist()
        print("Index persisted.")

    def load_and_ingest_documents(self) -> None:
        print("=== LOADING DOCUMENTS ===")
        documents = self.doc_loader.load()
        print(f"Loaded {len(documents)} raw documents.")

        if self.preprocessor:
            documents = self.preprocessor.preprocess(documents)
            print(f"Preprocessed to {len(documents)} documents.")

        if self.chunker:
            documents = self.chunker.chunk(documents)
            print(f"Chunked into {len(documents)} total chunks.")

        self.ingest_documents(documents)
