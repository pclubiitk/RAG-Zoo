from rag_src.post_retrival_enricher.base import PostBaseEnricher
from typing import List
import numpy as np


class SemanticFilter(PostBaseEnricher):
    """
    Filters out documents that are semantically dissimilar to the query
    using cosine similarity of embeddings.
    """

    def __init__(self, embedder, query_embedding, threshold: float = 0.75):
        """
        Args-
            embedder: Embedding model with embed(text: str) -> List[float]
            query_embedding: Precomputed embedding of the original query
            threshold: Minimum cosine similarity required to keep The doc
        """
        self.embedder = embedder
        self.query_embedding = query_embedding
        self.threshold = threshold

    def cosine_sim(self, v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def enrich(self, docs: List[str]) -> List[str]:
        filtered_docs = []
        for doc in docs:
            try:
                doc_emb = self.embedder.embed(doc)
                score = self.cosine_sim(doc_emb, self.query_embedding)

                if np.isnan(score):  # fallback for invalid cosine similarity
                    filtered_docs.append(doc)
                elif score >= self.threshold:
                    filtered_docs.append(doc)
            except Exception:
                filtered_docs.append(doc)  # fallback: keep doc if error

        return filtered_docs
