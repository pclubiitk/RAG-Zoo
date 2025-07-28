import os
import time
from dotenv import load_dotenv
from rag_src.llm import GeminiLLM
from rag_src.Complete_RAG_Pipeline.RunRAG import RunRAG
from rag_src.doc_loader.universal_doc_loader import UniversalDocLoader
from rag_src.query_transformer.multiquery import MultiQuery

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

docdir=r"C:\Users\harsh\Downloads\final_draft.pdf"
rag = RunRAG(
    llm=GeminiLLM(api_key=GEMINI_API_KEY),
    docdir=r"C:\Users\harsh\Downloads\final_draft.pdf",
    query_transform=MultiQuery,
    doc_loader=UniversalDocLoader(docdir),
)

async def run_rag_query(query: str) -> str:
    final_answer = []
    async for token in rag.run(query):
        final_answer.append(token)
        print(token, end="", flush=True)

    return "".join(final_answer)

async def test_rag_response():
    query = "What are the key differences between CBOW and Skip-Gram models in the context of word embeddings, and when is each preferred? How does the Transformer architecture improve upon the limitations of RNNs and LSTMs, particularly in handling long-range dependencies? In a Retrieval-Augmented Generation (RAG) pipeline, how do the roles of parametric memory (like BART) and non-parametric memory (like dense vector index) complement each other? How does Python’s asyncio help improve the performance of I/O-bound tasks in the ScyllaAgent chatbot, and what are the limitations compared to multiprocessing? What is the role of LangGraph in building modular AI workflows, and how does it integrate with LlamaIndex and Pydantic to ensure structured data handling?"
    
    start = time.perf_counter()  # ⏱️ Start timer

    answer = await run_rag_query(query)

    end = time.perf_counter()  # ⏱️ End timer
    elapsed = end - start

    print("\n\n📘 Final Answer:", answer.strip())
    print(f"⏱️ Time taken: {elapsed:.2f} seconds")

    # ✅ Sanity checks
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_rag_response())
