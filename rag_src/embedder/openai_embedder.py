from typing import List, Literal
from .base import BaseEmbedder
from llama_index.embeddings.openai import OpenAIEmbedding
import os


class OpenAIEmbedder(BaseEmbedder):
    """
    Embedder using OpenAI's embedding models via LlamaIndex.
    """

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str = None):
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Missing OpenAI API key. Set OPENAI_API_KEY in env or pass it explicitly."
            )

        self.embed_model = OpenAIEmbedding(model=model_name, api_key=self.api_key)

    def embed(
        self, texts: List[str], mode: Literal["query", "document"] = "document"
    ) -> List[List[float]]:
        return self.embed_model.get_text_embedding_batch(texts)
