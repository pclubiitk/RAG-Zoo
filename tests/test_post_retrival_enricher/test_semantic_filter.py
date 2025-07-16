import pytest
import random
from rag_src.post_retrival_enricher.semantic_filter import SemanticFilter


# Modified DummyEmbedder with no all-zero vectors
class DummyEmbedder:
    def embed(self, text):
        if "fail" in text:
            raise ValueError("Embedding error")
        elif "important" in text:
            return [1.0] * 10
        else:
            # return random non-zero low values
            return [random.uniform(0.01, 0.05) for _ in range(10)]


def test_semantic_filter_filters_below_threshold():
    embedder = DummyEmbedder()
    query_embedding = [1.0] * 10
    filterer = SemanticFilter(embedder, query_embedding, threshold=0.8)

    docs = ["this is important", "this is irrelevant"]
    result = filterer.enrich(docs)

    assert isinstance(result, list)
    assert "this is important" in result
    assert "this is irrelevant" not in result  # filtered out


def test_semantic_filter_keeps_all_above_threshold():
    embedder = DummyEmbedder()
    query_embedding = [1.0] * 10
    filterer = SemanticFilter(embedder, query_embedding, threshold=0.01)

    docs = ["this is important", "this is irrelevant"]
    result = filterer.enrich(docs)

    assert result == docs


def test_semantic_filter_fallback_on_error():
    embedder = DummyEmbedder()
    query_embedding = [1.0] * 10
    filterer = SemanticFilter(embedder, query_embedding, threshold=0.8)

    docs = ["fail to embed this one", "important doc"]
    result = filterer.enrich(docs)

    assert "fail to embed this one" in result  # fallback
    assert "important doc" in result


def test_cosine_similarity_computation():
    embedder = DummyEmbedder()
    filterer = SemanticFilter(embedder, [1.0, 0.0], threshold=0.5)
    score = filterer.cosine_sim([1.0, 0.0], [1.0, 0.0])

    assert pytest.approx(score) == 1.0


def test_semantic_filter_handles_zero_vector():
    class ZeroVectorEmbedder:
        def embed(self, text):
            if "zero" in text:
                return [0.0] * 10  # simulate bad embedding
            return [1.0] * 10  # valid

    query_embedding = [1.0] * 10
    filterer = SemanticFilter(ZeroVectorEmbedder(), query_embedding, threshold=0.5)

    docs = ["zero vector doc", "normal doc"]
    result = filterer.enrich(docs)

    assert "zero vector doc" in result  # fallback behavior
    assert "normal doc" in result
