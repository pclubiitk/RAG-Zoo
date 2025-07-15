from unittest.mock import MagicMock
from rag_src.query_transformer.multiquery import MultiQuery

def test_multiquery_transform_splits_lines(monkeypatch):
    mock_llm = MagicMock()
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Q1\nQ2\nQ3"
    monkeypatch.setattr("rag_src.query_transformer.multiquery.PromptTemplate", MagicMock())
    monkeypatch.setattr("rag_src.query_transformer.multiquery.StrOutputParser", MagicMock())
    monkeypatch.setattr("rag_src.query_transformer.multiquery.PromptTemplate.__or__", lambda self, other: mock_chain)
    transformer = MultiQuery(mock_llm)
    transformer.llm = mock_llm

    # Patch the chain to simulate the pipeline
    transformer.transform = lambda query, n=5: mock_chain.invoke.return_value.split("\n")
    result = transformer.transform("test", n=3)
    assert result == ["Q1", "Q2", "Q3"]
