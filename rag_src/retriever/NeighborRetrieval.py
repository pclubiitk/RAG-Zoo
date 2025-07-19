from .base import BaseRetriever
from typing import List, Dict, Any

class NeighborhoodContextRetriever(BaseRetriever):
    """
    A retriever that wraps around another BaseRetriever and augments results with neighboring context chunks.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,              # e.g., DefaultRetriever or any custom retriever
        all_documents: List[Dict[str, Any]],        # list of {"text": ..., "metadata": {"index": ...}}
        num_neighbors: int = 1,
        chunk_overlap: int = 20,
    ):
        super().__init__(top_k=base_retriever.top_k)
        self.base_retriever = base_retriever
        self.all_documents = all_documents
        self.num_neighbors = num_neighbors
        self.chunk_overlap = chunk_overlap

    async def retrieve(self, query: str) -> List[Dict[str, Any]]:
        top_chunks = self.base_retriever.retrieve(query)  # underlying semantic/keyword retrieval
        results = []

        for chunk in top_chunks:
            index = chunk["metadata"].get("index")
            if index is None:
                continue

            # grab neighbor window
            start = max(0, index - self.num_neighbors)
            end = index + self.num_neighbors + 1
            neighbors = [
                doc for doc in self.all_documents
                if start <= doc["metadata"]["index"] < end
            ]
            neighbors.sort(key=lambda x: x["metadata"]["index"])

            # concatenate with overlap logic
            combined = neighbors[0]["text"]
            for i in range(1, len(neighbors)):
                overlap_start = max(0, len(combined) - self.chunk_overlap)
            combined = combined[:overlap_start] + neighbors[i]["text"]

            results.append({
                "text": combined,
                "center_index": index,
                "neighbor_indices": [doc["metadata"]["index"] for doc in neighbors],
                "query_chunk": chunk["text"],
            })
        return results

