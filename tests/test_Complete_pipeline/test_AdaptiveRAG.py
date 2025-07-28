import os
import pytest
from dotenv import load_dotenv

from rag_src.Complete_RAG_Pipeline.AdaptiveRAG import AdaptiveRAG
from rag_src.doc_loader.universal_doc_loader import UniversalDocLoader
from rag_src.llm import GroqLLM
load_dotenv()

@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.path.exists(r"C:\Users\harsh\Downloads\final_draft.pdf"),
    reason="PDF document missing for test",
)
async def test_reliable_rag_groq_response():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    assert api_key is not None, "GROQ_API_KEY is missing in .env"

    # Initialize AdaptiveRAG with Groq LLM
    docdir=r"C:\Users\harsh\Downloads\FTS-376.pdf"
    rag_pipeline = AdaptiveRAG(
        llm=GroqLLM(api_key=api_key),
        docdir=docdir,
        doc_loader=UniversalDocLoader(docdir),
    )

    final_answer = []
    query = "What are the top 5 most consumed fruits in the U.S. according to the USDA?"
    async for token in rag_pipeline.run(query):
        final_answer.append(token)
        print(token, end="", flush=True)
        
    answer_str = "".join(final_answer)

    assert isinstance(answer_str, str), "Answer should be a string"
    assert len(answer_str.strip()) > 0, "Answer should not be empty"
    print("AdaptiveRAG Output:\n", answer_str)


