import os
import shutil
import pytest
import numpy as np
from rag_src.indexer import ChromaDBIndexer


@pytest.fixture
def dummy_data():
    embeddings = np.random.rand(3, 768).tolist()  # Match embedding_dim
    documents = ["Document one", "Document two", "Document three"]
    metadata = [{"source": "test1"}, {"source": "test2"}, {"source": "test3"}]
    return embeddings, documents, metadata


@pytest.fixture
def chroma_indexer(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("chroma_index")
    return ChromaDBIndexer(
        collection_name="test_collection",
        persist_directory=str(tmp_dir)
    )


def test_indexing_and_retrieval(chroma_indexer, dummy_data):
    embeddings, documents, metadata = dummy_data
    chroma_indexer.index(embeddings, documents, metadata)

    # Query all documents
    results = chroma_indexer.collection.get(include=["documents", "embeddings", "metadatas"])
    
    assert len(results["documents"]) == 3
    assert results["documents"][0] == "Document one"
    assert results["metadatas"][1]["source"] == "test2"
    assert len(results["embeddings"][0]) == 768


def test_reset_collection(chroma_indexer, dummy_data):
    embeddings, documents, metadata = dummy_data
    chroma_indexer.index(embeddings, documents, metadata)

    chroma_indexer.reset()
    results = chroma_indexer.collection.get(include=["documents"])
    assert results["documents"] == []


def test_persist(chroma_indexer, dummy_data):
    embeddings, documents, metadata = dummy_data
    chroma_indexer.index(embeddings, documents, metadata)
    chroma_indexer.persist()

    # Check that persistence directory has been created
    assert os.path.exists(chroma_indexer.persist_directory)

