# doc_preprocessor/

## `AdvancedPreprocessor`

Advanced preprocessor that performs lowercasing, stripping blank space, unicode normalization, HTML tag removal, special character cleanup

- Parameters (with default vals and types) for class object
    ```py
    * remove_stopwords: bool = False
    * language: str = "english"
    ```

- Functions:
    ```py
    - preprocess(self, docs: List[str]) -> List[str]
    ```
