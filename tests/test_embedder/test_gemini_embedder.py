import os
import pytest
from rag_src.embedder import GeminiEmbedder
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def api_key():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        pytest.skip("GOOGLE_API_KEY not set in environment")
    return key


@pytest.fixture
def embedder(api_key):
    return GeminiEmbedder(api_key=api_key)


def test_embedding_returns_vectors(embedder):
    texts = ["Gemini is powerful.", "Embeddings are fun."]
    embeddings = embedder.embed(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(vec, list) for vec in embeddings)
    assert all(len(vec) > 0 for vec in embeddings)


def test_embedding_modes(embedder):
    texts = ["This is a test sentence."]
    doc_embed = embedder.embed(texts, mode="document")
    query_embed = embedder.embed(texts, mode="query")

    assert isinstance(doc_embed, list)
    assert isinstance(query_embed, list)
    assert len(doc_embed[0]) == len(query_embed[0])  # Same model → same dim


def test_empty_input(embedder):
    embeddings = embedder.embed([])
    assert embeddings == []


def test_error_on_missing_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Missing Google API key"):
        GeminiEmbedder(api_key=None)
