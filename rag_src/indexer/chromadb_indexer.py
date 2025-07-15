from chromadb import PersistentClient
from typing import List, Optional, Dict, Any
from .base import BaseIndexer


class ChromaDBIndexer(BaseIndexer):
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_index",
    ):
        self.persist_directory = persist_directory
        self.client = PersistentClient(path=persist_directory)
        self.collection_name = collection_name

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, embedding_function=None
        )

    def index(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        ids = [f"doc_{i}" for i in range(len(documents))]
        metadatas = metadata if metadata else [{} for _ in range(len(documents))]

        self.collection.add(
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
        )

    def reset(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, embedding_function=None
        )

    def persist(self) -> None:
        # persistence is automatic in PersistentClient
        pass
