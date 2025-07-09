import os
import pytest
from rag_src.embedder import OpenAIEmbedder
from dotenv import load_dotenv
load_dotenv()
@pytest.fixture
def api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set in environment.")
    return key

@pytest.fixture
def embedder(api_key):
    return OpenAIEmbedder(model_name="text-embedding-3-small", api_key=api_key)

def test_embedding_output_shape(embedder):
    texts = ["This is a test.", "Embeddings from OpenAI."]
    embeddings = embedder.embed(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(vec, list) for vec in embeddings)
    assert all(len(vec) > 0 for vec in embeddings)

def test_query_vs_document_modes(embedder):
    texts = ["OpenAI rocks."]
    emb_doc = embedder.embed(texts, mode="document")
    emb_query = embedder.embed(texts, mode="query")

    assert isinstance(emb_doc, list)
    assert isinstance(emb_query, list)
    assert len(emb_doc[0]) == len(emb_query[0])  # Same model → same dimension

def test_empty_input(embedder):
    embeddings = embedder.embed([])
    assert embeddings == []

def test_error_on_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Missing OpenAI API key"):
        OpenAIEmbedder(api_key=None)
