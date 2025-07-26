# embedder/

## `OpenAIEmbedder`

- Parameters (with default vals and types) for class object:
    ```py
    * model_name: str = "text-embedding-3-small"
    * api_key: str = None
    ```

- Functions: 
    ```py
    - embed(
        self,
        texts: List[str],
        mode: Literal["query", "document"] = "document"
    ) -> List[List[float]]
    ```

## `GeminiEmbedder` 

Google GenAI multimodal embeddings

- Parameters (with default vals and types) for class object:
    ```py
    * model_name: str = "models/embedding-001"
    * api_key: str = None
    ```

- Functions: 
    ```py
    - embed(
        self,
        texts: List[str],
        mode: Literal["query", "document"] = "document"
    ) -> List[List[float]]
    ```
