from .base import BaseWebRetriever
from llama_index.core.schema import TextNode
from tavily import TavilyClient
from typing import List
import os


class TavilyWebRetriever(BaseWebRetriever):
    def __init__(self, api_key: str = None, max_results: int = 5):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY must be provided or set in environment")
        self.client = TavilyClient(api_key=self.api_key)
        self.max_results = max_results

    def retrieve(self, query: str) -> List[TextNode]:
        print(f"[TAVILY] Searching web for: {query}")
        try:
            results = self.client.search(query=query, max_results=self.max_results)
            nodes = []
            for res in results.get("results", []):
                text = f"{res.get('title', '')}\n{res.get('content', '')}".strip()
                url = res.get("url", "")
                if text:
                    nodes.append(TextNode(text=text, metadata={"source_url": url}))
            return nodes
        except Exception as e:
            print(f"[TAVILY ERROR] {e}")
            return [TextNode(text=f"Tavily search failed: {str(e)}")]
