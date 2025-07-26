# chunker/

## `RecursiveChunker` 

Hierarchical separators, overlap preserved  

- Parameters (with default vals and types) for class object:

    ```python
    * chunk_size: int = 512
    * chunk_overlap: int = 50
    ```

- Functions:

    ```py
    - chunk(self,docs: List[str],metadata: Optional[List[Dict[str, str]]] = None) -> List[str]
    ```

## `SemanticChunker`

Sentence-transformer similarity boundaries

- Parameters (with default vals and types) for class object:
    ```python
    * chunk_size: int = 512
    * chunk_overlap: int = 50
    ```

- Functions:
    ```py
    - chunk(self,docs: List[str],metadata: Optional[List[Dict[str, str]]] = None) -> List[str]
    ```

## `TokenChunker` 

Divides input texts into chunks with configurable overlaps

- Parameters (with default vals and types) for class object:
    ```python
    * chunk_size: int = 512
    * chunk_overlap: int = 50
    ```

- Functions:
    ```py
    - chunk(self,docs: List[str],metadata: Optional[List[Dict[str, str]]] = None) -> List[str]
    ```
