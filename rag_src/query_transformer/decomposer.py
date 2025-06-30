from typing import List
from llama_index.core.indices.query.query_transform.base import DecomposeQueryTransform
from rag_src.query_transformer.base import BaseQueryTransformer


class QueryDecomposer(BaseQueryTransformer):
    def __init__(self, llm, verbose: bool = False):
        self.transformer = DecomposeQueryTransform(llm=llm, verbose=verbose)

    def transform(self, query: str) -> List[str]:
        metadata = {"index_summary": "None"}  # dummy index summary for now
        bundle = self.transformer(query, metadata=metadata)
        return [bundle.query_str]  
