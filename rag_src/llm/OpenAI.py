from typing import List, Union
from llama_index.llms.openai import OpenAI
from .base import BaseLLM
import asyncio

class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4"):
        self.llm = OpenAI(model=model)

    def _generate_sync(self, query: str, contexts: List[str]) -> str:
        prompt = "\n\n".join(contexts) + "\n\n" + query
        return self.llm.complete(prompt).text

    async def generate(self, query: str, contexts: List[str]) -> str:
        return await asyncio.to_thread(self._generate_sync, query, contexts)
