# evaluator/

## `RelevanceEvaluator` 

Evaluates how relevant the generated response is to the original query using the provided contexts.

- Parameters (with default vals and types) for class object:
    ```py
    * llm: BaseLLM
    * threshold: float = 0.7
    ```

- Functions:
    ```py
    - evaluate(
        self,
        query: str,
        response: str,
        contexts: List[str]
    ) -> Dict[str, Any]
    ```

## `SegmentAttributor` 

Identifies exact segments from documents that support the generated answer

- Parameters (with default vals and types) for class object:
    ```py
    * llm
    ```

- Functions:
    ```py
    - locate_segments(
        self,
        query: str,
        response: str,
        docs: List[TextNode]
    ) -> List[Dict[str, Any]]
    - evaluate(
        self,
        query: str,
        response: str,
        contexts: List[str]
    ) -> Dict[str, Any]
    ```
