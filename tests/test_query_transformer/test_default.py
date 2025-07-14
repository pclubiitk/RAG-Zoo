from rag_src.query_transformer.default import DefaultQueryTransformer

def test_default_transform():
    transformer = DefaultQueryTransformer()
    query = "What is AI?"
    out = transformer.transform(query)
    assert out == [query]