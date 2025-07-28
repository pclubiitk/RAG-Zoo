import os
import pytest
from dotenv import load_dotenv

from rag_src.Complete_RAG_Pipeline.ReliableRAG import ReliableRAG
from rag_src.doc_loader.universal_doc_loader import UniversalDocLoader
from rag_src.llm import GroqLLM

@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.path.exists(r"C:\Users\harsh\Downloads\final_draft.pdf"),
    reason="PDF document missing for test"
)
async def test_reliable_rag_groq_response():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    assert api_key is not None, "GROQ_API_KEY is missing in .env"
    docdir=r"C:\Users\harsh\Downloads\final_draft.pdf"
    # Initialize ReliableRAG with Groq LLM
    rag = ReliableRAG(
        llm=GroqLLM(api_key=api_key),
        docdir=r"C:\Users\harsh\Downloads\final_draft.pdf",
        doc_loader=UniversalDocLoader(docdir),
    )
    final_answer=[]
    query = "What is Retrieval-Augmented Generation?"
    async for token in rag.run(query):
        final_answer.append(token)
        print(token, end="", flush=True)
        
    answer_str = "".join(final_answer)

    # Accept either string or .text based LLM outputs
    if hasattr(answer_str, "text"):
        answer_str = answer_str.text

    assert isinstance(answer_str, str), "Answer should be a string"
    assert len(answer_str.strip()) > 0, "Answer should not be empty"

    print("\n✅ ReliableRAG Output:\n", answer_str)
