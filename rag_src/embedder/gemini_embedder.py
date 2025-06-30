from typing import List, Literal
from .base import BaseEmbedder
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import os

class GeminiEmbedder(BaseEmbedder):
    """
    Gemini embedder using LlamaIndex's GoogleGenerativeAIEmbedding.
    """

    def __init__(self, model_name: str = "models/embedding-001", api_key: str = None):
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("Missing Google API key. Set GOOGLE_API_KEY env variable or pass it explicitly.")

        self.embed_model = GoogleGenAIEmbedding(
            model_name=model_name,
            api_key=self.api_key
        )

    def embed(
        self,
        texts: List[str],
        mode: Literal["query", "document"] = "document"
    ) -> List[List[float]]:
        return self.embed_model.get_text_embedding_batch(texts)
