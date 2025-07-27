# Complete RAG Pipeline

We'll see about the different modules in the Complete RAG Pipeline now...

## Module Summaries

### RunRAG
- Follows a path from question to answer by splitting documents, creating embeddings, indexing, and retrieving the best info.
- Parameters (with default vals and types) for class object:
    ```python
    * llm: Optional[BaseLLM]
    * embeddor: Optional[BaseEmbedder],
    * indexer: Optional[BaseIndexer]
    * retriever: Optional[BaseRetriever]
    * query_transform: Optional[BaseQueryTransformer]
    * doc_enricher: Optional[PreBaseEnricher]
    * doc_loader: Optional[BaseDocLoader]
    * preprocessor: Optional[BasePreprocessor]
    * docdir: str = "data"
    * chunker: Optional[BaseChunker] = None
    ```
- Functions:
    ```python
    # Check if FAISS index exists. If not, run the ingestion pipeline
    * ensure_index_ready(self)
    * run (self, query: str) -> str 
    * load_preprocess_chunk_documents(self) -> List[str]
    * ingest_documents(self, documents: List[str], metadata: Optional[List[dict]] = None) -> None
    ```

### AdaptiveRAG
- Changes its process depending on the type of question (like factual, opinion, or analytical).
- Parameters (with default vals and types) for class object:
    ```python
    * llm: Optional[BaseLLM] = None
    * embeddor: Optional[BaseEmbedder] = None,
    * indexer: Optional[BaseIndexer] = None
    * retriever: Optional[BaseRetriever] = None
    * preprocessor: Optional[BasePreprocessor] = None
    * chunker: Optional[BaseChunker] = None
    * doc_enricher: Optional[PostBaseEnricher] = None
    * docdir: str = "data"
    ```
- Functions:
    ```python
    # Factual queries: enhance the query for better retrieval
    * factual_retrieve(self, query) 
    * run (self, query: str) -> str 
    # Analytical queries: decompose into sub-questions, then rerank for diversity
    * analytical_retrieve(self, query, k=4) 
    # Opinion queries: extract perspectives, retrieve documents for each, then cluster
    * opinion_retrieve(self, query, k=4)
    # Contextual queries: include user context in reformulating the query
    * context_retrieve(self, query, k=4)
    # Index documents using embedding + indexing pipeline
    * ingest_documents(self, documents: List[str], metadata: Optional[List[dict]] = None) -> None
    # Load + preprocess + chunk + index documents (called if index is missing)
    * load_and_ingest_documents(self) -> None
    ```

### GraphRAG
- Uses a “knowledge graph” to see how facts and ideas connect.
- Parameters (with default vals and types) for class object:
    ```python
    * llm: Optional[BaseLLM]
    * embeddor: Optional[BaseEmbedder],
    * indexer: Optional[BaseIndexer]
    * retriever: Optional[BaseRetriever]
    * query_transform: Optional[BaseQueryTransformer]
    * doc_enricher: Optional[PreBaseEnricher]
    * doc_loader: Optional[BaseDocLoader]
    * preprocessor: Optional[BasePreprocessor]
    * docdir: str = "data"
    * chunker: Optional[BaseChunker] = None
    ```
- Functions:
    ```python
    # Maps user queries to Wikipedia article titles
    * infer_wikipedia_title(self, user_query: str) -> str 
    * run (self, query: str) -> str 
    * get_wikipedia_page(self, title: str) -> str
    * visualize_graph(self, output_file: str = "knowledge_graph.html")
    # Contextual queries: include user context in reformulating the query
    * context_retrieve(self, query, k=4)
    # Index documents using embedding + indexing pipeline
    * ingest_documents(self, documents: List[str], metadata: Optional[List[dict]] = None) -> None
    # Load + preprocess + chunk + index documents (called if index is missing)
    * load_and_ingest_documents(self) -> None
    ```

### CRAG (Corrective RAG)
-  It extends the standard RAG approach by dynamically evaluating and correcting the retrieval process, combining the power of vector databases, web search, and language models to provide accurate and context-aware responses to user queries
- Parameters (with default vals and types) for class object:
    ```python
    * llm: Optional[BaseLLM] = None
    * embeddor: Optional[BaseEmbedder] = None,
    * indexer: Optional[BaseIndexer] = None
    * retriever: Optional[BaseRetriever] = None
    * web_retriever: Optional[BaseWebRetriever] = None
    * evaluator: Optional[BaseEvaluator] = None
    * query_transform: Optional[BaseQueryTransformer] = None
    * doc_enricher: Optional[PostBaseEnricher] = None
    * doc_loader: Optional[BaseDocLoader] = None
    * preprocessor: Optional[BasePreprocessor] = None
    * chunker: Optional[BaseChunker] = None
    * docdir: str = "data"
    ```
- Functions:
    ```python
    * run (self, query: str) -> str
    ```

### ReliableRAG
- Makes sure answers are accurate and checks for halluncation. Gives highlighted document snippets
- Great when you need answers you can totally trust and prove.
- Parameters (with default vals and types) for class object:
    ```python
    * llm: Optional[BaseLLM] = None
    * embeddor: Optional[BaseEmbedder] = None
    * indexer: Optional[BaseIndexer] = None
    * retriever: Optional[BaseRetriever] = None
    * web_retriever: Optional[BaseWebRetriever] = None
    * query_transform: Optional[BaseQueryTransformer]
    * doc_loader: Optional[BaseDocLoader] = None
    * preprocessor: Optional[BasePreprocessor] = None
    * chunker: Optional[BaseChunker] = None
    * doc_enricher: Optional[PreBaseEnricher] = None
    * docdir: str = "data"
    ```
- Functions:
    ```python
    * run (self, query: str) -> str 
    * load_and_ingest_documents(self) -> None
    * ingest_documents(self, documents: List[str], metadata: Optional[List[dict]] = None) -> None
    ```

### RAPTOR
- Organizes info in a tree for very fast and smart searching.
- Parameters (with default vals and types) for class object:
    ```python
    * llm: Optional[BaseLLM] = None
    * embedder: Optional[BaseEmbedder] = None
    * indexer: Optional[BaseIndexer] = None
    * retriever: Optional[BaseRetriever] = None
    * web_retriever: Optional[BaseWebRetriever] = None
    * query_transform: Optional[BaseQueryTransformer] = None
    * doc_loader: Optional[BaseDocLoader] = None
    * preprocessor: Optional[BasePreprocessor] = None
    * chunker: Optional[BaseChunker] = None
    * doc_enricher: Optional[PreBaseEnricher] = None
    * docdir: str = "data"
    ```
- Functions:
    ```python
    * run(self, query: str, k: int = 3) -> str 
    * load_document(self)
    * extract_text(self, item)
    * embed_texts(self, texts: List[str]) -> List[List[float]]
    * perform_clustering(self, embeddings: np.ndarray, n_clusters: int = 10) -> np.ndarray
    * summarize_texts(self, texts: List[str]) -> str
    * create_retriever(self, vectorstore: FAISS) -> ContextualCompressionRetriever
    * build_vectorstore(self, tree_results: Dict[int, pd.DataFrame]) -> FAISS
    * build_raptor_tree(self) -> Dict[int, pd.DataFrame]
    ```

## Detailed Comparison Table

| Module       | Key Approach                             | How It Works                                                  | 
|--------------|-----------------------------------------|---------------------------------------------------------------|
| RunRAG       | Standard retrieval + generation          | Splits documents into chunks, converts to embeddings, builds index, retrieves most relevant chunks, and generates answer using AI. | 
| AdaptiveRAG  | Dynamic retrieval               | Classifies the question type (factual, analytical, opinion, contextual) and adjusts search and answer methods accordingly.   |
| GraphRAG     | Knowledge graph-based reasoning          | Picks best reference, extracts facts, builds a graph showing how facts connect, and uses this to answer.    | 
| CRAG         | Self-correcting   | Answers from internal sources first. If confidence is low,searches the web for fresh info | 
| ReliableRAG  | Verification and source attribution      | Produces answers with confidence scoring, double-checks facts to avoid hallucinations, clearly attributes information to sources. | 
| RAPTOR       | Hierarchical clustering/tree retrieval   | Breaks docs into layers of summaries (clusters), stores them in a tree structure. Goes into relevant clusters for efficient retrieval and answering. | 

---


