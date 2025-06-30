from typing import List, Dict, Any
from collections import defaultdict
import heapq
from base import BaseRetriever
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from llama_index.llms.ollama import Ollama

class FusionRetriever(BaseRetriever):
    def __init__(self, retriever: BaseRetriever, top_k: int = 5):
        super().__init__(top_k)
        self.retriever = retriever
        self.llm = Ollama(model = 'mistral')
        self.n_expansions = 3
    def expand_query(self, query: str) -> List[str]:
        prompt = f"""Rephrase the following search query in {self.n_expansions} different but semantically equivalent ways.
Each rephrasing should preserve the meaning but vary the wording and structure.

Query: "{query}"

List {self.n_expansions} variations, each on a new line.
"""

        response = self.llm.complete(prompt)
        lines = response.text.strip().split("\n")
        variants = [query]

        for line in lines:
            line = line.strip("-•1234567890. ").strip()
            if line and line.lower() != query.lower():
                variants.append(line)

        return list(set(variants))[:self.n_expansions + 1]
        

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        variants = self.expand_query(query)
        all_results = []

        for q in variants:
            results = self.retriever.retrieve(q)
            for r in results:
                r["source_query"] = q
                all_results.append(r)

        fused = {}
        for r in all_results:
            key = r["text"]
            if key not in fused:
                fused[key] = r
            else:
                fused[key]["score"] += r["score"]

        return sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:self.top_k]
