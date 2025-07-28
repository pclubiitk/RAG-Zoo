from typing import List
from rag_src.llm import BaseLLM
from rag_src.query_transformer.base import BaseQueryTransformer

class LLMWebQueryTransformer(BaseQueryTransformer):
    """
    Transforms a user query into a more web-search-optimized version using an LLM.
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def transform(self, query: str) -> List[str]:
        prompt = (
            "You are a helpful assistant. Rephrase the following query so it is optimized "
            "for a web search. Use keywords, simplify language, and remove ambiguity.\n\n"
            f"Original Query: {query}\n\n"
            "Web Search Query:"
        )

        transformed = self.llm.generate(prompt, contexts=[])
        return [transformed.strip()]
