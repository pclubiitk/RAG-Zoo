from rag_src.pre_embedding_enricher.base import PreBaseEnricher
from typing import List


class TopicTagger(PreBaseEnricher):
    """
    Uses the LLM to classify each document's topic.
    Appends the topic as a tag to the beginning of each doc.
    """

    def __init__(self, llm):
        self.llm = llm

    def enrich(self, docs: List[str]) -> List[str]:
        tagged_docs = []
        for doc in docs:
            prompt = f"Classify the main topic of the following document:\n\n{doc}"
            try:
                topic = self.llm.generate(prompt).strip().title()
            except:
                topic = "General"
            tagged_doc = f"[Topic: {topic}]\n\n{doc}"
            tagged_docs.append(tagged_doc)
        return tagged_docs
