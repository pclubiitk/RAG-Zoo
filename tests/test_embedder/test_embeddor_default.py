import pytest
from rag_src.embedder.default import DefaultEmbedder

@pytest.fixture
def embedder():
    return DefaultEmbedder("all-MiniLM-L6-v2")

def test_single_sentence_embedding(embedder):
    texts = ["Hello world."]
    embeddings = embedder.embed(texts)
    
    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    assert isinstance(embeddings[0], list)
    assert len(embeddings[0]) > 0  # Should be >0 dimension

def test_multiple_sentence_embeddings(embedder):
    texts = ["First sentence.", "Second sentence.", "Third one."]
    embeddings = embedder.embed(texts)
    
    assert len(embeddings) == len(texts)
    for emb in embeddings:
        assert isinstance(emb, list)
        assert len(emb) > 0

def test_mode_argument_does_not_break(embedder):
    texts = ["This is a query."]
    emb_query = embedder.embed(texts, mode="query")
    emb_doc = embedder.embed(texts, mode="document")
    
    # Should still return valid embeddings
    assert isinstance(emb_query, list)
    assert isinstance(emb_doc, list)
    assert len(emb_query[0]) == len(emb_doc[0])  # Same model, same vector size

def test_empty_input(embedder):
    embeddings = embedder.embed([])
    assert embeddings == []
