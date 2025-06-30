

import weaviate
from weaviate.util import get_valid_uuid
from typing import List, Optional, Dict, Any
from uuid import uuid4
from .base import BaseIndexer


class WeaviateIndexer(BaseIndexer):
    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        class_name: str = "DocumentChunk",
        recreate_schema: bool = True
    ):
        self.client = weaviate.Client(weaviate_url)
        self.class_name = class_name

        if recreate_schema and self.client.schema.contains({"classes": [{"class": self.class_name}]}):
            self.client.schema.delete_class(self.class_name)

        self._ensure_class()

    def _ensure_class(self):
        """
        Creates the schema class in Weaviate if it doesn't exist.
        """
        if not self.client.schema.contains({"classes": [{"class": self.class_name}]}):
            schema = {
                "class": self.class_name,
                "properties": [
                    {
                        "name": "text",
                        "dataType": ["text"]
                    },
                    {
                        "name": "metadata",
                        "dataType": ["text"]
                    }
                ],
                "vectorIndexType": "hnsw",
                "vectorizer": "none"
            }
            self.client.schema.create_class(schema)

    def index(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        for i, (vector, doc) in enumerate(zip(embeddings, documents)):
            meta = metadata[i] if metadata else {}
            self.client.data_object.create(
                data_object={
                    "text": doc,
                    "metadata": str(meta)
                },
                class_name=self.class_name,
                vector=vector,
                uuid=get_valid_uuid(str(uuid4()))
            )

    def reset(self) -> None:
        """
        Clears all documents in the Weaviate class.
        """
        if self.client.schema.contains({"classes": [{"class": self.class_name}]}):
            self.client.schema.delete_class(self.class_name)
            self._ensure_class()

    def persist(self) -> None:
        """
        Weaviate handles persistence internally.
        No-op for now.
        """
        print("[INFO] Persistence handled by Weaviate backend.")
