import pytest
from rag_src.post_retrival_enricher.default import PostDefaultEnricher

def test_post_default_enricher_returns_same_docs():
    enricher = PostDefaultEnricher()
    input_docs = [
        "Retrieved result 1",
        "Another chunk of context",
        "Final passage for retrieval"
    ]
    output_docs = enricher.enrich(input_docs)

    assert isinstance(output_docs, list)
    assert output_docs == input_docs
    assert all(isinstance(doc, str) for doc in output_docs)

def test_post_default_enricher_empty_input():
    enricher = PostDefaultEnricher()
    output_docs = enricher.enrich([])

    assert output_docs == []
