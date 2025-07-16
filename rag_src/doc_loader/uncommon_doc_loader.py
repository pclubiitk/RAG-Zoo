from typing import List, Union
from pathlib import Path
from bs4 import BeautifulSoup
import json
import csv

from ebooklib import epub

from rag_src.doc_loader import BaseDocLoader


class UncommonDocLoader(BaseDocLoader):
    """
    Loader for uncommon document formats: .epub, .csv, .json, .xml, .rst, .tex
    """

    SUPPORTED_EXTS = {".epub", ".csv", ".json", ".xml", ".rst", ".tex"}

    def __init__(self, path: Union[str, Path], encoding: str = "utf-8"):
        super().__init__(path)
        self.encoding = encoding

    def load(self) -> List[str]:
        paths = []

        if self.path.is_dir():
            for ext in self.SUPPORTED_EXTS:
                paths.extend(self.path.rglob(f"*{ext}"))
        elif self.path.is_file() and self.path.suffix.lower() in self.SUPPORTED_EXTS:
            paths.append(self.path)
        else:
            raise ValueError(f"Unsupported file or directory: {self.path}")

        contents = []
        for file_path in paths:
            ext = file_path.suffix.lower()
            try:
                if ext == ".epub":
                    contents.append(self._load_epub(file_path))
                elif ext == ".csv":
                    contents.append(self._load_csv(file_path))
                elif ext == ".json":
                    contents.append(self._load_json(file_path))
                elif ext == ".xml":
                    contents.append(self._load_xml(file_path))
                elif ext == ".rst" or ext == ".tex":
                    contents.append(self._load_plaintext(file_path))
            except Exception as e:
                print(f"[Warning] Skipping {file_path}: {e}")

        return contents

    def _load_epub(self, path: Path) -> str:
        book = epub.read_epub(str(path))
        text = []
        for item in book.get_items():
            if item.get_type() == epub.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                text.append(soup.get_text())
        return "\n".join(text)

    def _load_csv(self, path: Path) -> str:
        rows = []
        with open(path, "r", encoding=self.encoding) as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(", ".join(row))
        return "\n".join(rows)

    def _load_json(self, path: Path) -> str:
        with open(path, "r", encoding=self.encoding) as f:
            data = json.load(f)
        return json.dumps(data, indent=2)

    def _load_xml(self, path: Path) -> str:
        with open(path, "r", encoding=self.encoding) as f:
            soup = BeautifulSoup(f, "xml")  # use 'lxml-xml' parser for structured XML
        return soup.get_text(separator="\n")

    def _load_plaintext(self, path: Path) -> str:
        with open(path, "r", encoding=self.encoding) as f:
            return f.read()
