import pytest
from rag_src.post_retrival_enricher.doc_summarizer import DocSummarizer


class MockLLM:
    def generate(self, prompt):
        assert "Summarize the following document" in prompt
        return "This is a summary."


def test_doc_summarizer_basic():
    llm = MockLLM()
    enricher = DocSummarizer(llm=llm)

    docs = ["This is the full document text."]
    result = enricher.enrich(docs)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == "This is a summary."


def test_doc_summarizer_multiple_docs():
    llm = MockLLM()
    enricher = DocSummarizer(llm=llm)

    docs = ["Doc 1 content.", "Doc 2 content."]
    result = enricher.enrich(docs)

    assert len(result) == 2
    assert all(summary == "This is a summary." for summary in result)


def test_doc_summarizer_fallback_on_error():
    class FaultyLLM:
        def generate(self, prompt):
            raise RuntimeError("LLM failure")

    enricher = DocSummarizer(llm=FaultyLLM())
    docs = ["Doc A", "Doc B"]
    result = enricher.enrich(docs)

    assert result == docs  # fallback used


def test_doc_summarizer_empty_input():
    llm = MockLLM()
    enricher = DocSummarizer(llm=llm)

    docs = []
    result = enricher.enrich(docs)

    assert result == []


def test_doc_summarizer_with_llamaindex_mockllm():
    try:
        from llama_index.core.llms.mock import MockLLM as LlamaMockLLM
    except ImportError:
        pytest.skip("llama-index-core not installed")

    class MyMockLLM(LlamaMockLLM):
        def generate(self, prompt, **kwargs):
            return "This is a mock summary."

    llama_llm = MyMockLLM()
    enricher = DocSummarizer(llm=llama_llm)

    docs = ["Realistic input text"]
    result = enricher.enrich(docs)

    assert result == ["This is a mock summary."]
