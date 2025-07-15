from .base import BaseRetriever
from typing import List, Dict, Any
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer


class DefaultRetriever(BaseRetriever):
    """
    Default retriever using FAISS + SentenceTransformer.
    Loads an existing FAISS index and associated metadata/documents.
    """

    def __init__(
        self,
        index_path: str = "default_index",
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
    ):
        super().__init__(top_k)
        self.index_path = index_path
        self.model = SentenceTransformer(model_name)

        # Load FAISS index
        index_file = os.path.join(index_path, "index.faiss")
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"FAISS index not found at {index_file}")
        self.index = faiss.read_index(index_file)

        # Load documents and metadata
        with open(os.path.join(index_path, "data.pkl"), "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        # Encode the query
        query_vec = self.model.encode([query]).astype("float32")

        # Search the FAISS index
        distances, indices = self.index.search(query_vec, self.top_k)

        results = []
        for i in indices[0]:
            if i < len(self.documents):  # FAISS may return dummy indices if fewer items
                results.append(
                    {"text": self.documents[i], "metadata": self.metadata[i]}
                )

        return results
