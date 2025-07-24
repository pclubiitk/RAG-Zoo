import os
import pytest
from dotenv import load_dotenv

from rag_src.Complete_RAG_Pipeline.RunRAG import RunRAG
from rag_src.doc_loader.universal_doc_loader import UniversalDocLoader
from rag_src.llm.gemini import GeminiLLM
from rag_src.chunker import DefaultChunker

@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.path.exists(r"C:\Users\harsh\Downloads\FTS-376.pdf"),
    reason="Document file not found"
)
async def test_runrag_with_gemini():
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")
    assert gemini_key is not None, "GEMINI_API_KEY is not set in the environment"

    doc_path = r"C:\Users\harsh\Downloads\FTS-376.pdf"
    rag = RunRAG(
        llm=GeminiLLM(gemini_key),
        embeddor=None,
        indexer=None,
        retriever=None,
        query_transform=None,
        doc_enricher=None,
        doc_loader=UniversalDocLoader(doc_path),
        preprocessor=None,
        docdir=doc_path,
        chunker=DefaultChunker(chunk_size=512, chunk_overlap=50),
    )

    # rag.load_and_ingest_documents()
    final_answer = []
    query = "What are the top 5 most consumed fruits in the U.S. according to the USDA?"
    async for token in rag.run(query):
        final_answer.append(token)
        print(token, end="", flush=True)
        
    answer_str = "".join(final_answer)

    # Updated assertion for LLM output object
    assert isinstance(answer_str, str), "Expected answer to be a string"
    assert len(answer_str.strip()) > 0
    print("\n🧪 Gemini Output:\n", answer_str)
