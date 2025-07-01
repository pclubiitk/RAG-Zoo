from dotenv import load_dotenv
from rag_src.llm import GroqLLM
from rag_src.complete_RAG_Pipeline.CRAG import CRAG
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

crag = CRAG(llm=GroqLLM(api_key=GROQ_API_KEY), docdir=r"D:\data\final_draft.pdf")

def run_crag_query(query: str) -> str:
    return crag.run(query)

def test_crag_response():
    query = "Who wrote the song Loving you is a losing game?"
    answer = run_crag_query(query)
    
    # ✅ Basic test: check that the result is not empty
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
