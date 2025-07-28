from .base import BaseDocLoader
from typing import List, Union
from pathlib import Path


class DefaultDocLoader(BaseDocLoader):
    """
    Default document loader that reads .txt files from the given path.
    If a directory is given, it reads all .txt files inside it.
    If a file is given, it reads the content if it's a .txt file.
    """

    def __init__(self, path: Union[str, Path], encoding: str = "utf-8"):
        super().__init__(path)
        self.encoding = encoding

    def load(self) -> List[str]:
        if self.path.is_dir():
            txt_files = list(self.path.glob("*.txt"))
        elif self.path.is_file() and self.path.suffix == ".txt":
            txt_files = [self.path]
        else:
            raise ValueError(
                f"Invalid path: {self.path}. Must be a .txt file or directory containing .txt files."
            )

        contents = []
        for file_path in txt_files:
            with open(file_path, "r", encoding=self.encoding) as f:
                contents.append(f.read())

        return contents
