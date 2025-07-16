import pytest
from rag_src.pre_embedding_enricher.qa_pair_generator import QAPairGenerator


class MockLLM:
    def generate(self, prompt):
        if "error" in prompt:
            raise ValueError("Simulated LLM failure")
        return "Q: What is Python?\nA: A programming language."


def test_qa_pair_generator_basic_generation():
    docs = ["Python is a programming language."]
    enricher = QAPairGenerator(llm=MockLLM())
    enriched = enricher.enrich(docs)

    assert isinstance(enriched, list)
    assert len(enriched) == 1
    assert enriched[0].startswith("Q:")
    assert "A:" in enriched[0]


def test_qa_pair_generator_multiple_docs():
    docs = ["Doc1", "Doc2"]
    enricher = QAPairGenerator(llm=MockLLM())
    enriched = enricher.enrich(docs)

    assert len(enriched) == 2
    assert all("Q:" in e and "A:" in e for e in enriched)


def test_qa_pair_generator_handles_llm_failure():
    class FailingLLM:
        def generate(self, prompt):
            raise RuntimeError("Failed")

    docs = ["Fallback test document"]
    enricher = QAPairGenerator(llm=FailingLLM())
    enriched = enricher.enrich(docs)

    assert enriched[0] == docs[0]


def test_qa_pair_generator_mixed_results():
    class CustomLLM:
        def generate(self, prompt):
            if "Doc 2" in prompt:
                raise RuntimeError("fail")
            return "Q: Generated QA"

    docs = ["Doc 1", "Doc 2"]
    enricher = QAPairGenerator(llm=CustomLLM())
    enriched = enricher.enrich(docs)

    assert enriched[0] == "Q: Generated QA"
    assert enriched[1] == "Doc 2"


def test_qa_pair_generator_with_llamaindex_mockllm():
    try:
        from llama_index.core.llms.mock import MockLLM as LlamaMockLLM
    except ImportError:
        pytest.skip("llama-index-core not installed")

    class MyMockLLM(LlamaMockLLM):
        def generate(self, prompt, **kwargs):
            return "Q: What is RAG?\nA: A retrieval-augmented generation system."

    llama_llm = MyMockLLM()
    enricher = QAPairGenerator(llm=llama_llm)
    docs = ["RAG helps enhance LLMs with external knowledge."]
    enriched = enricher.enrich(docs)

    assert len(enriched) == 1
    assert enriched[0].startswith("Q:")
    assert "A:" in enriched[0]
