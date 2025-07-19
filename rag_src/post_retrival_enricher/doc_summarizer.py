from rag_src.post_retrival_enricher.base import PostBaseEnricher
from typing import List
import asyncio

class DocSummarizer(PostBaseEnricher):
    def __init__(self, llm):
        self.llm = llm  # llm.generate can be sync or async

    async def enrich(self, docs: List[str]) -> List[str]:
        async def summarize(doc: str) -> str:
            prompt = f"Summarize the following document in 1-2 lines:\n\n{doc}"
            try:
                if asyncio.iscoroutinefunction(self.llm.generate):
                    return await self.llm.generate(prompt)
                else:
                    return await asyncio.to_thread(self.llm.generate, prompt)
            except Exception:
                return doc  # fallback on failure

        return await asyncio.gather(*(summarize(doc) for doc in docs))
