import os
import time
from dotenv import load_dotenv
from rag_src.llm import GroqLLM
from rag_src.Complete_RAG_Pipeline.CRAG import CRAG
from rag_src.doc_loader.universal_doc_loader import UniversalDocLoader

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

docdir=r"C:\Users\harsh\Downloads\final_draft.pdf"
crag = CRAG(
    llm=GroqLLM(api_key=GROQ_API_KEY),
    docdir=r"C:\Users\harsh\Downloads\final_draft.pdf",
    doc_loader=UniversalDocLoader(docdir)
)

async def run_crag_query(query: str) -> str:
    final_answer = []
    async for token in crag.run(query):
        final_answer.append(token)
        print(token, end="", flush=True)

    return "".join(final_answer)

async def test_crag_response():
    query = "Who wrote the song Loving you is a losing game?"
    
    start = time.perf_counter()  # ⏱️ Start timer

    answer = await run_crag_query(query)

    end = time.perf_counter()  # ⏱️ End timer
    elapsed = end - start

    print("\n\n📘 Final Answer:", answer.strip())
    print(f"⏱️ Time taken: {elapsed:.2f} seconds")

    # ✅ Sanity checks
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_crag_response())
