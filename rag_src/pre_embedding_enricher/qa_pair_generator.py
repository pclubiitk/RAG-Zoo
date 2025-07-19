

from rag_src.pre_embedding_enricher.base  import PreBaseEnricher
from typing import List
import asyncio

class QAPairGenerator(PreBaseEnricher):
    """
    Converts documents into question–answer pairs using an LLM
    Helpful for improving grounding and context awareness.
    """

    def __init__(self, llm):
        self.llm = llm

    async def enrich(self, docs: List[str]) -> List[str]:
        async def generate_qa(doc: str) -> str:
            prompt = f"Convert the following document into 1-2 question-answer pairs:\n\n{doc}"
            try:
                qa = await self.llm.generate(prompt)
            except Exception:
                qa = doc  # fallback
            return qa

        # Run all QA generations concurrently
        qa_pairs = await asyncio.gather(*(generate_qa(doc) for doc in docs))
        return qa_pairs