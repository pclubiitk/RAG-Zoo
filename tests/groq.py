from rag_src.llm.gemini import GeminiLLM
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
llm = GeminiLLM(api_key=GEMINI_API_KEY)

reply = llm.generate(query="Write ", contexts= [])

print(reply)
