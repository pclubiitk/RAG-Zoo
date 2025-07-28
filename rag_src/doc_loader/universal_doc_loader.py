from typing import List, Union
from pathlib import Path
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import docx

from rag_src.doc_loader import BaseDocLoader  # Adjust import based on your structure


class UniversalDocLoader(BaseDocLoader):
    """
    Universal document loader supporting .txt, .md, .pdf, .docx, .html.
    """

    SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx", ".html", ".htm"}

    def __init__(self, path: Union[str, Path], encoding: str = "utf-8"):
        super().__init__(path)
        self.encoding = encoding

    def load(self) -> List[str]:
        paths = []

        if self.path.is_dir():
            # Get all supported files from the directory
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
                if ext in [".txt", ".md"]:
                    contents.append(self._load_txt(file_path))
                elif ext == ".pdf":
                    contents.append(self._load_pdf(file_path))
                elif ext == ".docx":
                    contents.append(self._load_docx(file_path))
                elif ext in [".html", ".htm"]:
                    contents.append(self._load_html(file_path))
            except Exception as e:
                print(f"[Warning] Skipping {file_path}: {e}")

        return contents

    def _load_txt(self, path: Path) -> str:
        with open(path, "r", encoding=self.encoding) as f:
            return f.read()

    def _load_pdf(self, path: Path) -> str:
        text = ""
        with fitz.open(path) as doc:
            for page in doc:
                text += page.get_text()
        return text

    def _load_docx(self, path: Path) -> str:
        doc = docx.Document(path)
        return "\n".join(para.text for para in doc.paragraphs)

    def _load_html(self, path: Path) -> str:
        with open(path, "r", encoding=self.encoding) as f:
            soup = BeautifulSoup(f, "html.parser")
            return soup.get_text(separator="\n")
