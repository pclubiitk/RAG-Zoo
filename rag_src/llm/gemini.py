from typing import List, Union
from llama_index.llms.google_genai import GoogleGenAI
from .base import BaseLLM

class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str = None, model: str = "gemini-1.5-flash"):
        self.llm = GoogleGenAI(model=model, api_key=api_key)

    def generate(self, query: str, contexts: List[str]) -> Union[str, dict]:
        prompt = "\n\n".join(contexts) + "\n\n" + query
        return self.llm.complete(prompt)
