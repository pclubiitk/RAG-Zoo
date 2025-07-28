from typing import List, Union
from llama_index.llms.ollama import Ollama
from rag_src.llm.base import BaseLLM
import asyncio

class OllamaLLM(BaseLLM):
    def __init__(self, model: str = "mistral"):
        self.llm = Ollama(model=model)

    def _generate_sync(self, query: str, contexts: List[str]) -> str:
        prompt = "\n\n".join(contexts) + "\n\n" + query
        return self.llm.complete(prompt).text

    async def generate(self, query: str, contexts: List[str]) -> str:
        return await asyncio.to_thread(self._generate_sync, query, contexts)
    
    async def generate_stream(self, query: str, contexts: List[str]):
        prompt = "\n\n".join(contexts) + "\n\n" + query
        resp = await self.llm.astream_complete(prompt)
        async for chunk in resp:
            yield chunk.text