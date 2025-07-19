from unittest.mock import MagicMock
from rag_src.query_transformer.decomposer import QueryDecomposer


class DummySubQ:
    def __init__(self, query_str):
        self.query_str = query_str


class DummyBundle:
    def __init__(self, subquestions=None, query_str=None):
        self.subquestions = subquestions
        self.query_str = query_str


def test_query_decomposer_with_subquestions():
    mock_llm = MagicMock()
    mock_transformer = MagicMock(
        return_value=DummyBundle(subquestions=[DummySubQ("a"), DummySubQ("b")])
    )
    decomposer = QueryDecomposer(mock_llm)
    decomposer.transformer = mock_transformer

    result = decomposer.transform("complex query")
    assert result == ["a", "b"]


def test_query_decomposer_without_subquestions():
    mock_llm = MagicMock()
    mock_transformer = MagicMock(
        return_value=DummyBundle(subquestions=None, query_str="main")
    )
    decomposer = QueryDecomposer(mock_llm)
    decomposer.transformer = mock_transformer

    result = decomposer.transform("complex query")
    assert result == ["main"]
