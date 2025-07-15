from unittest.mock import MagicMock


# Import the class to test
from rag_src.query_transformer.hyde import HyDe


class DummyBundle:
    def __init__(self, custom_embedding_strs=None, query_str=None):
        self.custom_embedding_strs = custom_embedding_strs
        self.query_str = query_str


def test_hyde_transform_returns_custom_embedding_strs():
    # Mock LLM and HyDEQueryTransform
    mock_llm = MagicMock()
    mock_transformer = MagicMock(
        return_value=DummyBundle(custom_embedding_strs=["foo", "bar"], query_str="baz")
    )

    # Patch HyDEQueryTransform in HyDe
    hyde = HyDe(mock_llm)
    hyde.transformer = mock_transformer

    result = hyde.transform("test query")
    assert result == ["foo", "bar"]


def test_hyde_transform_returns_query_str_when_no_custom_embedding_strs():
    mock_llm = MagicMock()
    mock_transformer = MagicMock(
        return_value=DummyBundle(custom_embedding_strs=None, query_str="baz")
    )
    hyde = HyDe(mock_llm)
    hyde.transformer = mock_transformer

    result = hyde.transform("test query")
    assert result == ["baz"]


def test_hyde_transform_calls_transformer_with_query():
    mock_llm = MagicMock()
    mock_transformer = MagicMock(
        return_value=DummyBundle(custom_embedding_strs=["foo"], query_str="baz")
    )
    hyde = HyDe(mock_llm)
    hyde.transformer = mock_transformer

    query = "some question"
    hyde.transform(query)
    mock_transformer.assert_called_once_with(query)
