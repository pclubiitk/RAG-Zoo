from unittest.mock import MagicMock, patch
from rag_src.retriever.FusionRetrieval import FusionRetriever


def test_fusion_retriever():
    # Mock base retriever
    base_retriever = MagicMock()
    base_retriever.retrieve.side_effect = [
        [{"text": "A", "metadata": {}, "score": 1.0}],  # Variant1
        [{"text": "B", "metadata": {}, "score": 2.0}],  # Variant2
        [{"text": "A", "metadata": {}, "score": 3.0}],  # Original query
    ]

    # Patch the LLM inside FusionRetriever to return mock variants
    with patch("llama_index.llms.ollama.Ollama.complete") as mock_complete:
        mock_complete.return_value = MagicMock(text="Variant1\nVariant2")

        # Create retriever with mocked LLM and base retriever
        fusion = FusionRetriever(base_retriever)

        # Run retrieval
        results = fusion.retrieve("query")

        # Assertions
        assert isinstance(results, list)
        texts = [r["text"] for r in results]
        assert "A" in texts
        assert "B" in texts
        assert len(results) >= 2  # Should include both A and B
