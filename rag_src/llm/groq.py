from typing import List, Union
from llama_index.llms.groq import Groq
import asyncio
from .base import BaseLLM

class GroqLLM(BaseLLM):
    def __init__(self, api_key: str = None, model: str = "llama3-8b-8192"):
        self.llm = Groq(model=model, api_key=api_key)

    def _generate_sync(self, query: str, contexts: List[str]) -> str:
        prompt = "\n\n".join(contexts) + "\n\n" + query
        return self.llm.complete(prompt).text

    async def generate(self, query: str, contexts: List[str]) -> str:
        return await asyncio.to_thread(self._generate_sync, query, contexts)