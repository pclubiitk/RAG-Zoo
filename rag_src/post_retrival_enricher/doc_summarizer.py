

from rag_src.post_retrival_enricher.base import PostBaseEnricher
from typing import List

class DocSummarizer(PostBaseEnricher):
    """
    This module summarizes each document using the LLM
    It helps reduce the size and the noise in the retrieved context.
    """

    def __init__(self, llm):
        self.llm = llm

    def enrich(self, docs: List[str]) -> List[str]:
        summaries = []
        for doc in docs:
            prompt = f"Summarize the following document in 1-2 lines:\n\n{doc}"
            try:
                summary = self.llm.generate(prompt)
            except:
                summary = doc  # fallback: keep original doc
            summaries.append(summary)
        return summaries