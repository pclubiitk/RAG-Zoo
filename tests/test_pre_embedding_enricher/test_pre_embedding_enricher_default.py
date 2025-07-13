import pytest
from rag_src.pre_embedding_enricher.default import PreDefaultEnricher

def test_pre_default_enricher_returns_same_docs():
    enricher = PreDefaultEnricher()
    input_docs = [
        "Document 1: LlamaIndex is a framework for LLM-powered applications.",
        "Document 2: RAG stands for Retrieval-Augmented Generation.",
    ]
    output_docs = enricher.enrich(input_docs)

    assert isinstance(output_docs, list)
    assert output_docs == input_docs
    assert all(isinstance(doc, str) for doc in output_docs)

def test_pre_default_enricher_empty_input():
    enricher = PreDefaultEnricher()
    output_docs = enricher.enrich([])
    assert output_docs == []
