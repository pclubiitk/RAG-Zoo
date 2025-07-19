from rag_src.retriever.NeighborRetrieval import NeighborhoodContextRetriever
from unittest.mock import MagicMock


def test_neighborhood_context_retriever():
    docs = [
        {"text": "A", "metadata": {"index": 0}},
        {"text": "B", "metadata": {"index": 1}},
        {"text": "C", "metadata": {"index": 2}},
    ]
    base = MagicMock()
    base.retrieve.return_value = [{"text": "B", "metadata": {"index": 1}}]
    retriever = NeighborhoodContextRetriever(
        base, docs, num_neighbors=1, chunk_overlap=0
    )
    results = retriever.retrieve("query")
    assert isinstance(results, list)
    assert "center_index" in results[0]
    assert results[0]["center_index"] == 1
    assert "neighbor_indices" in results[0]
    assert set(results[0]["neighbor_indices"]) == {0, 1, 2}
