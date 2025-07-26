# post_retrieval_enricher

## `DocSummarizer` 

It helps reduce the size and the noise in the retrieved context.

- Parameters (with default vals and types) for class object:
    ```python
    * llm
    ```

- Functions:
    ```py
    * enrich(self, docs: List[str]) -> List[str]
    ```

## `SelfRerank` 

This class re-ranks the documents using the LLM

- Parameters (with default vals and types) for class object:
    ```python
    * llm
    * top_k: int = 5
    ```

- Functions:
    ```py
    * enrich(self, docs: List[str]) -> List[str]
    ```

## `SemanticFilter` 

Filters out documents that are semantically dissimilar to the query using cosine similarity of embeddings.

- Parameters (with default vals and types) for class object:
    ```python
    * embedder
    * query_embedding
    * threshold: float = 0.75
    ```

- Functions:
    ```py
    * enrich(self, docs: List[str]) -> List[str]
    ```
