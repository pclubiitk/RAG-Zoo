

from rag_src.pre_embedding_enricher.base  import PreBaseEnricher
from typing import List
import asyncio

class TopicTagger(PreBaseEnricher):
    """
    Uses the LLM to classify each document's topic.
    Appends the topic as a tag to the beginning of each doc.
    """
    def __init__(self, llm):
        self.llm = llm

    async def enrich(self, docs: List[str]) -> List[str]:
        async def tag_document(doc: str) -> str:
            prompt = f"Classify the main topic of the following document:\n\n{doc}"
            try:
                topic = await self.llm.generate(prompt)
                topic = topic.strip().title()
            except Exception:
                topic = "General"
            return f"[Topic: {topic}]\n\n{doc}"

        # Run all tagging tasks concurrently
        tagged_docs = await asyncio.gather(*(tag_document(doc) for doc in docs))
        return tagged_docs