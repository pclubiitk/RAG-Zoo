
from .base import BaseContextEnricher
from typing import List

class SelfRerank(BaseContextEnricher):
    """
    This class re-ranks the documents using the LLM.
    For each document, we ask the LLM to rate its relevance 0 to 1,
    and then return the top-k documents with  highest score.
    """

    def __init__(self, llm, top_k: int = 5):
        self.llm = llm
        self.top_k = top_k

    def enrich(self, docs: List[str]) -> List[str]:
        ranked_docs = []

        for doc in docs:
            prompt = f"Give a relevance score (0 to 1) for the following doc:\n\n{doc}"
            try:
                score = float(self.llm.generate(prompt))
            except:
                score = 0.5  # default score if LLM fails
            ranked_docs.append((score, doc))

        # sort by score in descending order
        ranked_docs.sort(reverse=True, key=lambda x: x[0])

        # return only the document part from the top-k tuples
        return [doc for _, doc in ranked_docs[:self.top_k]]