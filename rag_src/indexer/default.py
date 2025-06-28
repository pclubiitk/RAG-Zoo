from .base import BaseIndexer
import faiss
import pickle
import os
from typing import List, Dict, Any, Optional
import numpy as np

class DefaultIndexer(BaseIndexer):
    """
    Default indexer using FAISS (Flat L2 index) and Pickle for metadata/doc persistence.
    """
    def __init__(self, persist_path: str = "default_index"):
        self.persist_path = persist_path
        self.index = None
        self.documents = []
        self.metadata = []

    def index(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        embeddings_np = np.array(embeddings).astype("float32")

        if self.index is None:
            dim = embeddings_np.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings_np)
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))

    def reset(self) -> None:
        self.index = None
        self.documents = []
        self.metadata = []

    def persist(self) -> None:
        os.makedirs(self.persist_path, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, os.path.join(self.persist_path, "index.faiss"))

        # Save documents and metadata
        with open(os.path.join(self.persist_path, "data.pkl"), "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadata": self.metadata
            }, f)