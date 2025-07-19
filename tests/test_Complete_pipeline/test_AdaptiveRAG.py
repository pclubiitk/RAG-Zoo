import os
import pytest
from dotenv import load_dotenv

from rag_src.Complete_RAG_Pipeline.AdaptiveRAG import AdaptiveRAG
from rag_src.doc_loader.universal_doc_loader import UniversalDocLoader
from rag_src.llm import GroqLLM
load_dotenv()
@pytest.mark.skipif(
    not os.path.exists(r"C:\Users\harsh\Downloads\final_draft.pdf"),
    reason="PDF document missing for test",
)
def test_reliable_rag_groq_response():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    assert api_key is not None, "GROQ_API_KEY is missing in .env"

    # Initialize AdaptiveRAG with Groq LLM
    docdir=r"C:\Users\DELL\Downloads\final_draft.pdf"
    rag_pipeline = AdaptiveRAG(
        llm=GroqLLM(api_key=api_key),
        docdir=docdir,
        doc_loader=UniversalDocLoader(docdir),
    )

    query = "What is RAG? How is it used? Also what is Abstract Factory Method?"
    answer = rag_pipeline.run(query)

    # Accept either string or .text based LLM outputs
    if hasattr(answer, "text"):
        answer = answer.text

    assert isinstance(answer, str), "Answer should be a string"
    assert len(answer.strip()) > 0, "Answer should not be empty"
    print("AdaptiveRAG Output:\n", answer)


