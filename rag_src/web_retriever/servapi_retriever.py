from .base import BaseWebRetriever
from llama_index.core.schema import TextNode
from serpapi import GoogleSearch
from typing import List
import os
import asyncio


class SerpAPIWebRetriever(BaseWebRetriever):
    def __init__(self, api_key: str = None, max_results: int = 5):
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY must be provided or set in environment.")
        self.max_results = max_results

    async def retrieve(self, query: str) -> List[TextNode]:
        print(f"[SERPAPI] Searching web for: {query}")
        try:
            params = {
                "q": query,
                "api_key": self.api_key,
                "engine": "google",
                "num": self.max_results,
            }

            search = GoogleSearch(params)
            results = await asyncio.to_thread(search.get_dict())

            web_nodes = []

            for res in results.get("organic_results", []):
                title = res.get("title", "")
                snippet = res.get("snippet", "")
                url = res.get("link", "")

                if snippet:
                    content = f"{title}\n{snippet}"
                    web_nodes.append(
                        TextNode(text=content, metadata={"source_url": url})
                    )

            print(f"[SERPAPI] Retrieved {len(web_nodes)} results.")
            return web_nodes

        except Exception as e:
            print(f"[SERPAPI ERROR] {e}")
            return [TextNode(text=f"SerpAPI search failed: {str(e)}")]
