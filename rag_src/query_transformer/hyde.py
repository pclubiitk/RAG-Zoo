from typing import List
from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
from rag_src.query_transformer.base import BaseQueryTransformer


class HyDe(BaseQueryTransformer):
    def __init__(self, llm, include_original: bool = True):
        self.llm = llm
        self.transformer = HyDEQueryTransform(llm=llm, include_original=include_original)

    def transform(self, query: str) -> List[str]:
        bundle = self.transformer(query)
        return bundle.custom_embedding_strs or [bundle.query_str]
