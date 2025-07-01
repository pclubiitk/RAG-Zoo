from .base import BaseWebRetriever
from llama_index.core.schema import TextNode
from duckduckgo_search import DDGS
from typing import List

class DuckDuckGoWebRetriever(BaseWebRetriever):
    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def retrieve(self, query: str) -> List[TextNode]:
        print(f"[DUCKDUCKGO] Searching web for: {query}")
        nodes = []
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=self.max_results)

                for res in results:
                    title = res.get("title", "")
                    snippet = res.get("body", "")
                    url = res.get("href", "")

                    if snippet:
                        content = f"{title}\n{snippet}"
                        nodes.append(TextNode(text=content, metadata={"source_url": url}))

            print(f"[DUCKDUCKGO] Retrieved {len(nodes)} results.")
            return nodes

        except Exception as e:
            print(f"[DUCKDUCKGO ERROR] {e}")
            return [TextNode(text=f"DuckDuckGo search failed: {str(e)}")]
