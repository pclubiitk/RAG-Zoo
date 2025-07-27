# doc_loader/

## `UncommonDocLoader` 

Doc loader for uncommon document formats: .epub, .csv, .json, .xml, .rst, .tex

- Parameters (with default vals and types) for class object:
    ```python
    * path: Union[str, Path]
    * encoding: str = "utf-8"
    ```

- Functions:
    ```python
    - load(self) -> List[str]
    - _load_epub(self, path: Path) -> str
    - _load_csv(self, path: Path) -> str
    - _load_json(self, path: Path) -> str
    - _load_xml(self, path: Path) -> str
    - _load_plaintext(self, path: Path) -> str
    ```

## `UniversalDocLoader` 

Universal document loader supporting .txt, .md, .pdf, .docx, .html

- Parameters (with default vals and types) for class object:
    ```py
    * path: Union[str, Path]
    * encoding: str = "utf-8"
    ```

- Functions:
    ```py
    * load(self) -> List[str]
    * _load_txt(self, path: Path) -> str
    * _load_pdf(self, path: Path) -> str
    * _load_docx(self, path: Path) -> str
    * _load_html(self, path: Path) -> str
    ```
