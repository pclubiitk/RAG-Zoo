from typing import List, Union
from llama_index.llms.google_genai import GoogleGenAI
import asyncio
from .base import BaseLLM

class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str = None, model: str = "gemini-1.5-flash"):
        self.llm = GoogleGenAI(model=model, api_key=api_key)

    def _generate_sync(self, query: str, contexts: List[str]) -> str:
        prompt = "\n\n".join(contexts) + "\n\n" + query
        return str(self.llm.complete(prompt))

    async def generate(self, query: str, contexts: List[str]) -> str:
        return await asyncio.to_thread(self._generate_sync, query, contexts)
