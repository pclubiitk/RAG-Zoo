from typing import List
from llama_index.core.indices.query.query_transform.base import DecomposeQueryTransform
from rag_src.query_transformer.base import BaseQueryTransformer


class QueryDecomposer(BaseQueryTransformer):
    def __init__(self, llm, verbose: bool = False):
        self.transformer = DecomposeQueryTransform(llm=llm, verbose=verbose)

    def transform(self, query: str) -> List[str]:
        metadata = {"index_summary": "None"}  # dummy index summary for now
        bundles = self.transformer(query, metadata=metadata)
        if hasattr(bundles, "subquestions") and bundles.subquestions is not None:
            return [sq.query_str for sq in bundles.subquestions]
        else:
            return [bundles.query_str]
    

