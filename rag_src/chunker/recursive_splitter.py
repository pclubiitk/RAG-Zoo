from typing import List, Dict, Optional
from .base import BaseChunker

from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.core.schema import Document


class RecursiveCharacterTextSplitter(BaseChunker):
    """
    Chunker using Langchain's RecursiveCharacterTextSplitter via LangchainNodeParser.
    Preserves structure and supports metadata prefixing.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.parser = LangchainNodeParser(self.text_splitter)

    def chunk(
        self,
        docs: List[str],
        metadata: Optional[List[Dict[str, str]]] = None
    ) -> List[str]:

        # Prepare documents for LlamaIndex
        doc_objs = []
        for i, doc in enumerate(docs):
            meta = metadata[i] if metadata and i < len(metadata) else {}
            doc_objs.append(Document(text=doc, metadata=meta))

        # Use Langchain splitter
        nodes = self.parser.get_nodes_from_documents(doc_objs)

        # Extract text (with metadata already embedded by LlamaIndex)
        return [node.text for node in nodes]
