from typing import List, Dict, Optional
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
)  # ✅ Requires: pip install llama-index-embeddings-huggingface
from rag_src.chunker.base import BaseChunker


class SemanticChunker(BaseChunker):
    """
    Chunker using LlamaIndex's SemanticSplitterNodeParser.
    Leverages token-aware, semantic splitting for high-quality chunks.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)

        # ✅ Required embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.parser = SemanticSplitterNodeParser(
            embed_model=self.embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(
        self, docs: List[str], metadata: Optional[List[Dict[str, str]]] = None
    ) -> List[str]:
        """Split docs into semantically meaningful chunks."""
        all_chunks = []

        for i, doc in enumerate(docs):
            doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
            llama_doc = Document(text=doc, metadata=doc_metadata)

            nodes = self.parser.get_nodes_from_documents([llama_doc])
            for node in nodes:
                text = node.text
                prefix = " | ".join(f"{k}: {v}" for k, v in node.metadata.items())
                if prefix:
                    all_chunks.append(f"{prefix} | {text}")
                else:
                    all_chunks.append(text)

        return all_chunks
