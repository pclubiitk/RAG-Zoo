import pytest
from unittest.mock import Mock, patch
from rag_src.retriever.explainable import ExplainableRetriever


@pytest.fixture
def mock_base_retriever():
    """
    Mocks a BaseRetriever that returns dummy document chunks.
    """
    mock = Mock()
    mock.retrieve.return_value = [
        {"text": "The sun emits light and heat."},
        {"text": "Earth rotates around the sun."}
    ]
    return mock


@patch("rag_src.retriever.explainable.Ollama")
def test_explainable_retriever(mock_ollama_class, mock_base_retriever):
    """
    Ensures ExplainableRetriever adds explanations to retrieved results using mocked LLM.
    """
    # Setup: mock LLM to return a dummy explanation
    mock_llm = Mock()
    mock_llm.complete.return_value.text = "Because the document talks about sunlight."
    mock_ollama_class.return_value = mock_llm

    # Create retriever with mock base retriever and mocked Ollama
    retriever = ExplainableRetriever(retriever=mock_base_retriever, top_k=2)
    results = retriever.retrieve("Why is sunlight important?")

    # Assertions
    assert isinstance(results, list)
    assert len(results) == 2
    for result in results:
        assert "text" in result
        assert "explanation" in result
        assert "sunlight" in result["explanation"].lower()

    # Ensure dependencies were called
    mock_base_retriever.retrieve.assert_called_once()
    assert mock_llm.complete.call_count == 2
