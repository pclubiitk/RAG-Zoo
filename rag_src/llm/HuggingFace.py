from typing import List, Union
from llama_index.llms.huggingface import HuggingFaceLLM
from .base import BaseLLM
import asyncio

class HuggingFaceLLMWrapper(BaseLLM):
    def __init__(self, model_name: str = "HuggingFaceH4/zephyr-7b-alpha"):
        self.llm = HuggingFaceLLM(model_name=model_name)

    def _generate_sync(self, query: str, contexts: List[str]) -> str:
        prompt = "\n\n".join(contexts) + "\n\n" + query
        return self.llm.complete(prompt).text

    async def generate(self, query: str, contexts: List[str]) -> str:
        return await asyncio.to_thread(self._generate_sync, query, contexts)
