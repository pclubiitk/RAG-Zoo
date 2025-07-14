import os
import pytest
from rag_src.llm.groq import GroqLLM
from dotenv import load_dotenv
load_dotenv()
@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set in environment"
)
def test_groqllm_real_api():
    llm = GroqLLM(api_key=os.getenv("GROQ_API_KEY"))
    result = llm.generate("What is the capital of France?", ["Geography context."])
    
    assert isinstance(result, str)
    assert "Paris" in result or len(result) > 0  # Very basic check
