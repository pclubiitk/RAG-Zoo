from .base import BaseWebRetriever
from .tavily import TavilyWebRetriever
from .servapi import SerpAPIWebRetriever
from .duckduckgo import DuckDuckGoWebRetriever

from llama_index.core.schema import TextNode
from typing import List, Optional, Set


class HybridWebRetriever(BaseWebRetriever):
    def __init__(
        self,
        tavily: Optional[BaseWebRetriever] = None,
        serpapi: Optional[BaseWebRetriever] = None,
        duckduckgo: Optional[BaseWebRetriever] = None,
        max_results: int = 5,
    ):
        self.tavily = tavily or TavilyWebRetriever(max_results=max_results)
        self.serpapi = serpapi or SerpAPIWebRetriever(max_results=max_results)
        self.duckduckgo = duckduckgo or DuckDuckGoWebRetriever(max_results=max_results)

    def _deduplicate(self, nodes: List[TextNode]) -> List[TextNode]:
        seen: Set[str] = set()
        unique_nodes = []
        for node in nodes:
            url = node.metadata.get("source_url", "")
            if url and url not in seen:
                unique_nodes.append(node)
                seen.add(url)
        return unique_nodes

    def retrieve(self, query: str) -> List[TextNode]:
        print(f"[HYBRID] Attempting web search for: {query}")

        for name, retriever in [
            ("Tavily", self.tavily),
            ("SerpAPI", self.serpapi),
            ("DuckDuckGo", self.duckduckgo),
        ]:
            try:
                results = retriever.retrieve(query)
                results = self._deduplicate(results)

                if results:
                    print(f"[HYBRID] Using {name} results ({len(results)} results)")
                    return results
            except Exception as e:
                print(f"[HYBRID] {name} failed: {e}")

        print("[HYBRID] All retrievers failed. Returning empty result.")
        return [TextNode(text="All web search providers failed.")]
