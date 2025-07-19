from rag_src.pre_embedding_enricher.base import PreBaseEnricher
from typing import List


class QAPairGenerator(PreBaseEnricher):
    """
    Converts documents into question–answer pairs using an LLM
    Helpful for improving grounding and context awareness.
    """

    def __init__(self, llm):
        self.llm = llm

    def enrich(self, docs: List[str]) -> List[str]:
        qa_pairs = []
        for doc in docs:
            prompt = f"Convert the following document into 1-2 question-answer pairs:\n\n{doc}"
            try:
                qa = self.llm.generate(prompt)
            except:
                qa = doc  # fallback to original
            qa_pairs.append(qa)
        return qa_pairs
