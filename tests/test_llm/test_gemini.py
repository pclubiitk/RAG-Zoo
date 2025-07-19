import pytest
import os
from rag_src.llm.gemini import GeminiLLM
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(scope="module")
def llm():
    # Load API key from env var or hardcode (not recommended)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not set. Skipping real API test.")
    return GeminiLLM(api_key=api_key, model="gemini-1.5-flash")


def test_generate_basic_response(llm):
    query = "What is the capital of Japan?"
    contexts = ["Japan is an island nation in East Asia."]
    response = llm.generate(query, contexts)

    assert isinstance(response, str)
    assert len(response.strip()) > 0


def test_generate_with_empty_context(llm):
    query = "Who wrote Hamlet?"
    contexts = []
    response = llm.generate(query, contexts)

    assert isinstance(response, str)
    assert len(response.strip()) > 0


def test_generate_multiple_contexts(llm):
    query = "Explain the process of photosynthesis."
    contexts = [
        "Photosynthesis occurs in the chloroplasts of plant cells.",
        "It converts light energy into chemical energy.",
        "Carbon dioxide and water are converted into glucose and oxygen.",
    ]
    response = llm.generate(query, contexts)

    assert isinstance(response, str)
    assert len(response.strip()) > 0
