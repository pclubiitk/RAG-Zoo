# rag_pipeline/indexer/chromadb_indexer.py

from chromadb import Client
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from typing import List, Optional, Dict, Any
from .base import BaseIndexer


class ChromaDBIndexer(BaseIndexer):
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_index",
        embedding_dim: int = 768  # Set this to match your embedding model
    ):
        self.settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory
        )
        self.client = Client(self.settings)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None  # We're manually providing embeddings
        )

    def index(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        ids = [f"doc_{i}" for i in range(len(documents))]
        metadatas = metadata if metadata else [{} for _ in range(len(documents))]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def reset(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None
        )

    def persist(self) -> None:
        self.client.persist()
