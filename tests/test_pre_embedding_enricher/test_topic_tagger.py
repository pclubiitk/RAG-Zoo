import pytest
from rag_src.pre_embedding_enricher.topic_tagger import TopicTagger


class MockLLM:
    def generate(self, prompt):
        if "error" in prompt:
            raise ValueError("Simulated failure")
        return "technology"


def test_topic_tagger_basic_tagging():
    docs = ["AI is transforming the world."]
    enricher = TopicTagger(llm=MockLLM())
    enriched = enricher.enrich(docs)

    assert len(enriched) == 1
    assert enriched[0].startswith("[Topic: Technology]")
    assert "AI is transforming the world." in enriched[0]
    assert isinstance(enriched[0], str)


def test_topic_tagger_multiple_docs():
    docs = ["Doc 1 about space.", "Doc 2 about biology."]
    enricher = TopicTagger(llm=MockLLM())
    enriched = enricher.enrich(docs)

    assert len(enriched) == 2
    assert all(doc.startswith("[Topic: Technology]") for doc in enriched)


def test_topic_tagger_handles_failure():
    class FailingLLM:
        def generate(self, prompt):
            raise RuntimeError("LLM crashed")

    docs = ["Should fallback to General topic"]
    enricher = TopicTagger(llm=FailingLLM())
    enriched = enricher.enrich(docs)

    assert enriched[0].startswith("[Topic: General]")
    assert "Should fallback to General topic" in enriched[0]


def test_topic_tagger_with_llamaindex_mockllm():
    try:
        from llama_index.core.llms.mock import MockLLM as LlamaMockLLM
    except ImportError:
        pytest.skip("llama-index-core not installed")

    llama_llm = LlamaMockLLM(response="finance")
    enricher = TopicTagger(llm=llama_llm)
    docs = ["Stock markets fluctuate due to investor behavior."]
    enriched = enricher.enrich(docs)

    assert len(enriched) == 1
    assert enriched[0].startswith("[Topic: Finance]")
    assert "Stock markets" in enriched[0]
