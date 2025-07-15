from unittest.mock import MagicMock
from rag_src.query_transformer.web_query_transformer import LLMWebQueryTransformer

def test_llm_web_query_transformer_generates(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "optimized query"
    transformer = LLMWebQueryTransformer(mock_llm)
    query = "Original query"
    result = transformer.transform(query)
    assert result == ["optimized query"]
    mock_llm.generate.assert_called_once()
