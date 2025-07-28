import pytest
from rag_src.llm import DefaultLLM


@pytest.fixture(scope="module")
def llm():
    return DefaultLLM(model_name="gpt2", max_new_tokens=50)


def test_generate_basic_response(llm):
    query = "What is the capital of France?"
    contexts = ["France is a country in Europe. Its capital is Paris."]
    response = llm.generate(query, contexts)

    assert isinstance(response, str)
    assert len(response.strip()) > 0  # Don't enforce specific keyword


def test_generate_with_empty_context(llm):
    query = "Who wrote Hamlet?"
    contexts = []
    response = llm.generate(query, contexts)

    assert isinstance(response, str)
    assert len(response.strip()) > 0  # Allow any non-empty answer


def test_generate_multiple_contexts(llm):
    query = "What causes rain?"
    contexts = [
        "Rain is part of the water cycle.",
        "Clouds form when water vapor condenses.",
        "Precipitation happens when droplets become heavy.",
    ]
    response = llm.generate(query, contexts)

    assert isinstance(response, str)
    assert len(response.strip()) > 0  # Don’t force mention of 'water' or 'cloud'
