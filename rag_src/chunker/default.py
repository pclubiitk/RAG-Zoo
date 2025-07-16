from typing import List, Dict, Optional
from rag_src.chunker.base import BaseChunker
from llama_index.core.text_splitter import SentenceSplitter


class DefaultChunker(BaseChunker):
    """
    Default chunker using LlamaIndex's SentenceSplitter.
    Applies metadata as prefix to each chunk.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self.splitter = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def chunk(
        self, docs: List[str], metadata: Optional[List[Dict[str, str]]] = None
    ) -> List[str]:

        all_chunks = []

        for i, doc in enumerate(docs):
            prefix = ""
            if metadata and i < len(metadata):
                meta = metadata[i]
                prefix = " | ".join(f"{k}: {v}" for k, v in meta.items())
                if prefix:
                    prefix += " | "

            # Split the document using SentenceSplitter
            chunks = self.splitter.split_text(doc)

            # Prepend metadata to each chunk
            enriched_chunks = [prefix + chunk for chunk in chunks]
            all_chunks.extend(enriched_chunks)

        return all_chunks
