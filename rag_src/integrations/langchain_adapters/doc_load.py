from typing import List, Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from rag_src.doc_loader.base import BaseDocLoader


class LangChainLoader(BaseLoader):
    """
    An adapter class that wraps any RAG-ZOO loader (which inherits from BaseDocLoader)
    and makes it compatible with the LangChain ecosystem.
    """

    def __init__(self, ragzoo_loader: BaseDocLoader):
        """
        Initializes the adapter with an instance of a RAG-ZOO loader.
        Args:
            ragzoo_loader: An instance of a class that inherits from your BaseDocLoader
                         (e.g., DefaultDocLoader, UniversalDocLoader).
        """
        self.ragzoo_loader = ragzoo_loader

    def load(self) -> List[Document]:
        """
        Loads documents using the wrapped RAG-ZOO loader and converts
        them into a list of LangChain Document objects.
        Returns:
            A list of LangChain Documents.
        """
        string_contents = self.ragzoo_loader.load()

        documents = []
        for content in string_contents:
            metadata = {"source": str(self.ragzoo_loader.path)}
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        return documents

    def lazy_load(self) -> Iterator[Document]:
        """
        Loads documents lazily, yielding one LangChain Document at a time.
        This is more memory-efficient for very large numbers of documents.
        """
        string_contents = self.ragzoo_loader.load()
        metadata = {"source": str(self.ragzoo_loader.path)}

        for content in string_contents:
            yield Document(page_content=content, metadata=metadata)
