from typing import List, Literal
from sentence_transformers import SentenceTransformer
from .base import BaseEmbedder

class DefaultEmbedder(BaseEmbedder):
    """
    Default embedder using SentenceTransformer.
    Treats both queries and documents identically (can customize if needed).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.model = SentenceTransformer(model_name)

    def embed(
        self,
        texts: List[str],
        mode: Literal["query", "document"] = "document"
    ) -> List[List[float]]:
        # Mode is currently not used differently, but could be in future.
        embeddings = self.model.encode(texts, convert_to_numpy=False)
        return [emb.tolist() for emb in embeddings]
