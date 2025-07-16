# tests/test_self_rerank.py

import pytest
from rag_src.post_retrival_enricher.self_rerank import SelfRerank


class MockLLM:
    def __init__(self, responses):
        self.responses = responses
        self.call_idx = 0

    def generate(self, prompt):
        response = self.responses[self.call_idx]
        self.call_idx += 1
        return response


def test_self_rerank_basic_topk2():
    docs = ["doc A", "doc B", "doc C"]
    llm = MockLLM(["0.2", "0.9", "0.6"])
    reranker = SelfRerank(llm=llm, top_k=2)

    result = reranker.enrich(docs)

    assert result == ["doc B", "doc C"]  # sorted by scores
    assert len(result) == 2


def test_self_rerank_fallback_on_llm_failure():
    class MixedLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            responses = ["0.3", "0.7", "0.5"]
            if self.calls < len(responses):
                result = responses[self.calls]
            else:
                raise ValueError("Simulated LLM failure")
            self.calls += 1
            return result

    docs = ["doc A", "doc B", "doc C", "doc D"]
    reranker = SelfRerank(llm=MixedLLM(), top_k=2)

    result = reranker.enrich(docs)

    # doc B (0.7), doc C or D (both 0.5, one from LLM, one fallback)
    assert len(result) == 2
    assert "doc B" in result
    assert any(doc in result for doc in ["doc C", "doc D"])


def test_self_rerank_less_than_topk():
    docs = ["only one doc"]
    llm = MockLLM(["0.8"])
    reranker = SelfRerank(llm=llm, top_k=5)

    result = reranker.enrich(docs)

    assert result == ["only one doc"]
    assert len(result) == 1


def test_self_rerank_empty_input():
    reranker = SelfRerank(llm=MockLLM([]), top_k=3)
    result = reranker.enrich([])

    assert result == []


def test_self_rerank_with_llamaindex_mockllm():
    try:
        from llama_index.core.llms.mock import MockLLM as LlamaMockLLM
    except ImportError:
        pytest.skip("llama-index-core not installed")

    docs = ["doc A", "doc B", "doc C"]
    llama_llm = LlamaMockLLM(response="0.7")  # same score for all
    reranker = SelfRerank(llm=llama_llm, top_k=2)

    result = reranker.enrich(docs)

    assert len(result) == 2
    assert all(doc in docs for doc in result)
