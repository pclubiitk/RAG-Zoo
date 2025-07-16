import pytest
from rag_src.post_retrival_enricher.semantic_filter import SemanticFilter
import numpy as np


class DummyEmbedder:
    def embed(self, text):
        if "fail" in text:
            raise ValueError("Embedding error")
        elif "important" in text:
            return [1.0, 0.0]  # High similarity
        elif "irrelevant" in text:
            return [0.01, 1.0]  # Low similarity
        else:
            return [0.5, 0.5]  # Neutral


def test_semantic_filter_filters_below_threshold():
    embedder = DummyEmbedder()
    query_embedding = [1.0, 0.0]
    filterer = SemanticFilter(embedder, query_embedding, threshold=0.6)

    docs = ["this is important", "this is irrelevant"]
    result = filterer.enrich(docs)

    assert "this is important" in result
    assert "this is irrelevant" not in result


def test_semantic_filter_keeps_all_above_threshold():
    embedder = DummyEmbedder()
    query_embedding = [1.0, 0.0]
    filterer = SemanticFilter(embedder, query_embedding, threshold=0.001)

    docs = ["this is important", "this is irrelevant"]
    result = filterer.enrich(docs)

    assert sorted(result) == sorted(docs)


def test_semantic_filter_fallback_on_error():
    embedder = DummyEmbedder()
    query_embedding = [1.0, 0.0]
    filterer = SemanticFilter(embedder, query_embedding, threshold=0.8)

    docs = ["fail to embed this one", "important doc"]
    result = filterer.enrich(docs)

    assert "fail to embed this one" in result  # fallback on error
    assert "important doc" in result


def test_cosine_similarity_computation():
    embedder = DummyEmbedder()
    filterer = SemanticFilter(embedder, [1.0, 0.0], threshold=0.5)

    score = filterer.cosine_sim([1.0, 0.0], [1.0, 0.0])
    assert pytest.approx(score, 0.001) == 1.0


def test_semantic_filter_handles_zero_vector():
    class ZeroVectorEmbedder:
        def embed(self, text):
            if "zero" in text:
                return [0.0, 0.0]  # simulate invalid vector (causes NaN)
            return [1.0, 0.0]

    query_embedding = [1.0, 0.0]
    filterer = SemanticFilter(ZeroVectorEmbedder(), query_embedding, threshold=0.5)

    docs = ["zero vector doc", "normal doc"]
    result = filterer.enrich(docs)

    assert "zero vector doc" in result  # fallback on nan
    assert "normal doc" in result
