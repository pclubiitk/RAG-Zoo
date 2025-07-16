from unittest.mock import patch, MagicMock
import pytest


# Mock the CRAG and GroqLLM components
@patch("rag_src.Complete_RAG_Pipeline.CRAG")
@patch("rag_src.llm.GroqLLM")
def test_crag_response(mock_groqllm, mock_crag_class):
    # Create a mock CRAG instance with a fake .run() method
    mock_crag_instance = MagicMock()
    mock_crag_instance.run.return_value = "A fictional artist wrote the song."

    # Replace the class return value with our mock instance
    mock_crag_class.return_value = mock_crag_instance

    # Optional: also mock GroqLLM (if it has behavior you're concerned about)
    mock_groqllm.return_value = MagicMock()

    # Create CRAG with mocked GroqLLM
    crag = mock_crag_class(mock_groqllm(api_key="fake-key"), docdir="fake/path.pdf")

    # Run the function under test
    query = "Who wrote the song Loving you is a losing game?"
    answer = crag.run(query)

    # ✅ Assertions
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
