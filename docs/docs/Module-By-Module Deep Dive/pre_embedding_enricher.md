# pre_embedding_enricher/

## `MetadataInjector` 

Adds metadata (like title, author, timestamp) to each document. Metadata is provided as a dictionary indexed by document position

- Parameters (with default vals and types) for class object:
    ```python
    * metadata: dict
    ```

- Functions:
    ```py
    * enrich(self, docs: List[str]) -> List[str]
    ```

## `QAPairGenerator` 

Converts documents into question–answer pairs using an LLM. Helpful for improving grounding and context awareness

- Parameters (with default vals and types) for class object:
    ```python
    * llm
    ```

- Functions:
    ```py
    * enrich(self, docs: List[str]) -> List[str]
    ```

## `TopicTagger` 

Uses the LLM to classify each document's topic.Appends the topic as a tag to the beginning of each doc.

- Parameters (with default vals and types) for class object:
    ```python
    * llm
    ```

- Functions:
    ```py
    * enrich(self, docs: List[str]) -> List[str]
    ```
