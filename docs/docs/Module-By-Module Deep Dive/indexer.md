# indexer/

## `ChromaDBIndexer` 

- Parameters (with default vals and types) for class object:
    ```python
    * collection_name: str = "rag_documents"
    * persist_directory: str = "./chroma_index"
    ```

- Functions
    ```py
    - index(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None
    - reset(self) -> None
    - persist(self) -> None
    ```

## `WeaviateIndexer` 

- Parameters (with default vals and types) for class object:
    ```python
    * weaviate_url: str
    * api_key: Optional[str] = None
    * class_name: str = "DocumentChunk"
    * recreate_schema: bool = True
    ```

- Functions
    ```py
    - index(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None
    - reset(self) -> None
    - persist(self) -> None
    ```

## `FAISS` is the default indexer