from typing import List, Optional
import os

from rag_src.llm import SmartLLM
from rag_src.retriever import BaseRetriever, DefaultRetriever
from rag_src.embedder import BaseEmbedder, DefaultEmbedder
from rag_src.query_transformer import BaseQueryTransformer, DefaultQueryTransformer
from rag_src.pre_embedding_enricher import PreBaseEnricher, PreDefaultEnricher
from rag_src.indexer import BaseIndexer, DefaultIndexer
from rag_src.doc_loader import BaseDocLoader, DefaultDocLoader
from rag_src.doc_preprocessor import BasePreprocessor, DefaultPreprocessor
from rag_src.chunker import BaseChunker, DefaultChunker

from llama_index.core import Document, KnowledgeGraphIndex
from llama_index.core.graph_stores.simple import SimpleGraphStore
from pyvis.network import Network
import wikipedia


class GraphRAG:
    def __init__(
        self,
        llm: Optional[SmartLLM],
        embedder: Optional[BaseEmbedder],
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
        self.llm = llm or SmartLLM()
        self.embedder = embedder or DefaultEmbedder()
        self.indexer = indexer or DefaultIndexer()
        self.query_transform = query_transform or DefaultQueryTransformer()
        self.doc_enricher = doc_enricher or PreDefaultEnricher()
        self.doc_loader = doc_loader or DefaultDocLoader(self.docdir)
        self.preprocessor = preprocessor or DefaultPreprocessor()
        self.chunker = chunker or DefaultChunker()

        self.graph_store = SimpleGraphStore()

        # Ensure index is built before initializing retriever
        index_path = getattr(self.indexer, "persist_path", "default_index")
        index_file = os.path.join(index_path, "index.faiss")

        if not os.path.exists(index_file):
            print(
                f"[INFO] FAISS index not found at {index_file}. Running ingestion pipeline."
            )
            self.load_and_ingest_documents()
        else:
            print(
                f"[INFO] Found existing index at {index_file}. Skipping ingestion.")

        self.retriever = retriever or DefaultRetriever(index_path=index_path)
        self.kg_index = None

    def infer_wikipedia_title(self, user_query: str) -> str:
        prompt = (
            "You are an assistant that maps user queries to Wikipedia article titles.\n"
            "Return ONLY the title of the most relevant Wikipedia article for the query below.\n"
            "Do NOT include any explanations or additional text. Output must be just the title.\n"
            f"Query: {user_query}\n"
            "Title:"
        )
        title = self.llm.generate(prompt, [])

        # Post-processing to clean up LLM verbosity
        title = title.strip().strip('"').strip()
        title = title.split("\n")[0]  # Only first line
        title = title.replace("Title:", "").strip()

        # Fallback: if LLM still returns bad title
        if not title or len(title.split()) > 10 or ":" in title:
            print(
                f"[WARNING] LLM returned suspicious title: {title}. Falling back to search."
            )
            results = wikipedia.search(user_query)
            if not results:
                raise ValueError(
                    f"No fallback Wikipedia results found for: {user_query}"
                )
                title = results[0]

        print(f"[GraphRAG] Using Wikipedia page title: '{title}'")
        return title

    def run(self, query: str) -> str:
        print("=== RUNNING GRAPHRAG PIPELINE ===")

        wikipedia.set_lang("en")
        page_title = self.infer_wikipedia_title(query)

        try:
            text = self.get_wikipedia_page(page_title)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch '{page_title}': {e}")

        docs = [Document(text=text)]

        self.kg_index = KnowledgeGraphIndex.from_documents(
            docs,
            graph_store=self.graph_store,
            max_triplets_per_chunk=10,
            include_embeddings=False,
            llm=self.llm,
            embed_model=self.embedder,
        )

        graph_query = (
            f"Using the knowledge graph built from '{page_title}', "
            f"answer the following: {query}"
        )

        query_engine = self.kg_index.as_query_engine(llm=self.llm)
        resp = query_engine.query(graph_query)

        print("=== Graph Query Result ===")
        print(resp.response)
        return resp.response

    def get_wikipedia_page(self, title: str) -> str:
        try:
            return wikipedia.page(title, auto_suggest=False).content
        except wikipedia.DisambiguationError as e:
            return wikipedia.page(e.options[0]).content
        except wikipedia.PageError:
            search_results = wikipedia.search(title)
            if not search_results:
                raise ValueError(f"No results found for '{title}'")
            return wikipedia.page(search_results[0]).content

    def visualize_graph(self, output_file: str = "knowledge_graph.html"):
        if self.kg_index is None:
            raise ValueError("Knowledge Graph not built yet. Call run() first")

        triplets = self.kg_index.get_triples()
        net = Network(
            height="750px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            notebook=False,
            directed=True,
        )

        for subj, pred, obj in triplets:
            net.add_node(subj, label=subj, color="#03DAC6")
            net.add_node(obj, label=obj, color="#BB86FC")
            net.add_edge(subj, obj, title=pred, label=pred, color="white")

        net.repulsion(node_distance=120, central_gravity=0.3)
        net.show(output_file)
        print(f"Graph saved to: {output_file}")

    def ingest_documents(
        self, documents: List[str], metadata: Optional[List[dict]] = None
    ) -> None:
        if not self.embedder or not self.indexer:
            raise ValueError("Embedder or indexer not set.")

        print("=== INDEXING DOCUMENTS ===")
        embeddings = self.embedder.embed(documents)
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
