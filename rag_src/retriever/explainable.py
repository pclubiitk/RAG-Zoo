from typing import List, Dict, Any
from .base import BaseRetriever
from llama_index.llms.ollama import Ollama
import asyncio

class ExplainableRetriever(BaseRetriever):
    def __init__(self, retriever: BaseRetriever, top_k: int = 5):
        super().__init__(top_k)
        self.retriever = retriever
        self.llm = Ollama(model='mistral')  # or any other supported model

    async def generate_explanation(self, query: str, document: str) -> str:
        prompt = f"""You are an AI assistant helping explain search results.

Query: "{query}"

Document: "{document}"

Explain clearly and concisely why this document is relevant to the query.
"""
        response = await self.llm.acomplete(prompt)
        return response.text.strip()

    async def retrieve(self, query: str) -> List[Dict[str, Any]]:
        results = self.retriever.retrieve(query)
        
        tasks = [
            self.generate_explanation(query, result["text"])
            for result in results[:self.top_k]
        ]
        explanations = await asyncio.gather(*tasks)

        for result, explanation in zip(results, explanations):
            result["explanation"] = explanation

        return results[:self.top_k]
