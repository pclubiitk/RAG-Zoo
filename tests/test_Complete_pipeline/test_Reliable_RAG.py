import os
import pytest
from dotenv import load_dotenv

from rag_src.Complete_RAG_Pipeline.ReliableRAG import ReliableRAG
from rag_src.llm import GroqLLM


@pytest.mark.skipif(
    not os.path.exists("D:/data/final_draft.pdf"),
    reason="PDF document missing for test",
)
def test_reliable_rag_groq_response():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    assert api_key is not None, "GROQ_API_KEY is missing in .env"

    # Initialize ReliableRAG with Groq LLM
    rag = ReliableRAG(llm=GroqLLM(api_key=api_key), docdir="D:/data/final_draft.pdf")

    query = "What is Retrieval-Augmented Generation?"
    answer = rag.run(query)

    # Accept either string or .text based LLM outputs
    if hasattr(answer, "text"):
        answer = answer.text

    assert isinstance(answer, str), "Answer should be a string"
    assert len(answer.strip()) > 0, "Answer should not be empty"

    print("\n✅ ReliableRAG Output:\n", answer)
