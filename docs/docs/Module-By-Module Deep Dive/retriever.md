# retriever/

## `FusionRetriever` 

Retrieves and gives a combined set of top documents from multiple retrieval methods or query variants.

- Parameters (with default vals and types) for class object:
    ```python
    * retriever: BaseRetriever
    * top_k: int = 5
    ```

- Functions:
    ```py
    * expand_query(self, query: str) -> List[str]
    * retrieve(self, query: str) -> List[Dict[str, Any]]
    ```

## `ReRankingRetriever` 

Retrieves and gives top k documents

- Parameters (with default vals and types) for class object:
    ```python
    * index_path = "default_index"
    * model_name = "all-MiniLM-L6-v2"
    * reranker_model_name="BAAI/bge-reranker-large"
    * initial_top_n:int = 20
    ```

- Functions:
    ```py
    * rerank(self, query: str, docs:List[Dict[str, Any]], k:int =5) -> List[Dict[str, Any]]
    * retrieve(self, query:str, k:int =5) -> List[Dict[str, Any]]
    ```

## `NeighborhoodContextRetriever` 

Graph-aware context expansion

- Parameters (with default vals and types) for class object:
    ```python
    * base_retriever: BaseRetriever
    * all_documents: List[Dict[str, Any]]
    * num_neighbors: int = 1
    * chunk_overlap: int = 20
    ```

- Functions:
    ```py
    * retrieve(self, query: str) -> List[Dict[str, Any]]  
    ```

## `ExplainableRetriever` 

Retrieves and gives token-limited text chunks from documents, optionally enriched with metadata.

- Parameters (with default vals and types) for class object:
    ```python
    * retriever: BaseRetriever
    * top_k: int = 5
    ```

- Functions:
    ```py
    * generate_explanation(self, query: str, document: str) -> str
    * retrieve(self, query: str) -> List[Dict[str, Any]]
    ```

---

# web_retriever/

## `DuckDuckGoWebRetriever` 

Retrieves stuff via DuckDuckGo

- Parameters (with default vals and types) for class object:
    ```python
    * max_results: int = 5
    ```

- Functions:
    ```py
    * retrieve(self, query: str) -> List[TextNode]
    ```

## `SerpAPIWebRetriever` 

Retrieves stuff via SerpAPI

- Parameters (with default vals and types) for class object:
    ```python
    * api_key: str = None
    * max_results: int = 5
    ```

- Functions:
    ```py
    * retrieve(self, query: str) -> List[TextNode]
    ```

## `TavilyWebRetriever` 

Retrieves stuff via TavilyAPI

- Parameters (with default vals and types) for class object:
    ```python
    * api_key: str = None
    * max_results: int = 5
    ```

- Functions:
    ```py
    * retrieve(self, query: str) -> List[TextNode]
    ```

## `HybridWebRetriever` 

Attempts retrieval via a combination of the above three

- Parameters (with default vals and types) for class object:
    ```python
    * tavily: Optional[BaseWebRetriever] = None
    * serpapi: Optional[BaseWebRetriever] = None
    * duckduckgo: Optional[BaseWebRetriever] = None
    * max_results: int = 5
    ```

- Functions:
    ```py
    * _deduplicate(self, nodes: List[TextNode]) -> List[TextNode]
    * retrieve(self, query: str) -> List[TextNode]
    ```
