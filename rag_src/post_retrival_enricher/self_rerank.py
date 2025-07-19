from rag_src.post_retrival_enricher.base import PostBaseEnricher
from typing import List
import asyncio

class SelfRerank(PostBaseEnricher):
    """
    This class re-ranks the documents using the LLM.
    For each document, we ask the LLM to rate its relevance 0 to 1,
    and then return the top-k documents with  highest score.
    """

    def __init__(self, llm, top_k: int = 5):
        self.llm = llm
        self.top_k = top_k
        
    async def _get_score(self, doc: str) -> float:
        prompt = f"Give a relevance score (0 to 1) for the following doc:\n\n{doc}"
        try:
            if self._is_async:
                result = await self.llm.generate(prompt)
            else:
                result = await asyncio.to_thread(self.llm.generate, prompt)
            return float(result)
        except:
            return 0.5,doc
        
    async def enrich(self, docs: List[str]) -> List[str]:
        ranked_docs = await asyncio.gather(*[self._get_score(doc) for doc in docs])
        # sort by score in descending order
        ranked_docs.sort(reverse=True, key=lambda x: x[0])

        # return only the document part from the top-k tuples
        return [doc for _, doc in ranked_docs[:self.top_k]]