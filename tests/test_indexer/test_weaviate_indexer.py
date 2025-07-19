import pytest
import random
import os
from dotenv import load_dotenv

from rag_src.indexer.weaviate_indexer import WeaviateIndexer

load_dotenv()

WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")


@pytest.fixture(scope="module")
def weaviate_indexer():
    return WeaviateIndexer(
        weaviate_url=WEAVIATE_URL,
        api_key=WEAVIATE_API_KEY,
        class_name="CloudDocTest",
        recreate_schema=True,
    )


@pytest.fixture
def dummy_data():
    embeddings = [[random.random() for _ in range(5)] for _ in range(3)]
    documents = ["Cloud document 1", "Cloud document 2", "Cloud document 3"]
    metadata = [{"source": "cloud1"}, {"source": "cloud2"}, {"source": "cloud3"}]
    return embeddings, documents, metadata


def test_index_cloud_data(weaviate_indexer, dummy_data):
    embeddings, documents, metadata = dummy_data
    weaviate_indexer.index(embeddings, documents, metadata)

    # No exception means indexing likely succeeded
    assert True


def test_reset_cloud_schema(weaviate_indexer):
    weaviate_indexer.reset()

    # Use newer client.collections.list_all() for v4 compatibility
    collection_names = weaviate_indexer.client.collections.list_all()
    assert weaviate_indexer.class_name in collection_names


def test_cloud_persist_noop(weaviate_indexer):
    weaviate_indexer.persist()
    assert True
