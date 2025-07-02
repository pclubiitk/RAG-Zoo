import os
import pytest
from dotenv import load_dotenv

from rag_src.llm import SmartLLM  # or your custom LLM
from rag_src.Complete_RAG_Pipeline.GraphRAG import GraphRAG

# Optionally skip if no internet for Wikipedia
@pytest.mark.skipif("CI" in os.environ, reason="Skipping test in CI due to no internet access")
def test_graphrag_with_gemini():
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    assert groq_key is not None, "GROQ_API_KEY not set in .env"

    # Create instance of GraphRAG with GeminiLLM and default components
    rag = GraphRAG(
        llm=SmartLLM(),
        embedder=None,
        indexer=None,
        retriever=None,
        query_transform=None,
        doc_enricher=None,
        doc_loader=None,
        preprocessor=None,
        docdir=r"D:\data\final_draft.pdf",  # or any path for doc loading (won't be used in Wikipedia mode)
        chunker=None,
    )

    query = "Who discovered the law of gravity?"
    response = rag.run(query)

    assert isinstance(response, str), "Output is not a string"
    assert len(response.strip()) > 0, "Response is empty"
    print(f"GraphRAG response: {response}")
