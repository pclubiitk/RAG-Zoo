
from rag_src.post_retrival_enricher.base import PostBaseEnricher
from typing import List
import numpy as np
import asyncio

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

    async def enrich(self, docs: List[str]) -> List[str]:
        async def score_doc(doc):
            try:
                doc_emb = await asyncio.to_thread(self.embedder.embed, [doc])
                score = self.cosine_sim(doc_emb[0], self.query_embedding)
                if score >= self.threshold:
                    return doc
            except:
                return doc  # fallback
            return None

        results = await asyncio.gather(*[score_doc(doc) for doc in docs])
        return [doc for doc in results if doc]