from uuid import uuid4
from typing import List, Optional, Dict, Any

from weaviate import connect_to_weaviate_cloud
from weaviate.collections import Collection
from weaviate.collections.classes.config import Configure, Property, DataType
from weaviate.classes.init import Auth
from weaviate.collections.classes.data import DataObject
from rag_src.indexer import BaseIndexer


class WeaviateIndexer(BaseIndexer):
    def __init__(
        self,
        weaviate_url: str,
        api_key: Optional[str] = None,
        class_name: str = "DocumentChunk",
        recreate_schema: bool = True
    ):
        self.client = connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(api_key) if api_key else None
        )
        self.class_name = class_name

        if recreate_schema and self.client.collections.exists(self.class_name):
            self.client.collections.delete(name=self.class_name)

        if not self.client.collections.exists(self.class_name):
            self.client.collections.create(
                name=self.class_name,
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="metadata", data_type=DataType.TEXT),
                ],
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=Configure.VectorIndex.hnsw()
            )

        self.collection: Collection = self.client.collections.get(self.class_name)

    def index(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        objs = []
        for i, (embedding, text) in enumerate(zip(embeddings, documents)):
            obj = DataObject(
                uuid=str(uuid4()),
                properties={
                    "text": text,
                    "metadata": str(metadata[i]) if metadata else "{}"
                },
                vector=embedding
            )
            objs.append(obj)

        self.collection.data.insert_many(objs)
        
    def reset(self) -> None:
        if self.client.collections.exists(self.class_name):
            self.client.collections.delete(name=self.class_name)

        self.client.collections.create(
            name=self.class_name,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="metadata", data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw()
        )
        self.collection = self.client.collections.get(self.class_name)

    def persist(self) -> None:
        print("[INFO] Weaviate handles persistence automatically.")
