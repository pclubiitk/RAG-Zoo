import os
import shutil
import pickle
import numpy as np
import pytest
from rag_src.indexer import DefaultIndexer


@pytest.fixture
def temp_index_dir():
    path = "tests/temp_index"
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)


@pytest.fixture
def dummy_data():
    embeddings = np.random.rand(3, 5).tolist()  # 3 vectors of dimension 5
    documents = ["doc1", "doc2", "doc3"]
    metadata = [{"id": 1}, {"id": 2}, {"id": 3}]
    return embeddings, documents, metadata


def test_index_and_reset(dummy_data):
    embeddings, documents, metadata = dummy_data
    indexer = DefaultIndexer()

    indexer.index(embeddings, documents, metadata)
    assert indexer.faiss_index.ntotal == 3
    assert len(indexer.documents) == 3
    assert indexer.metadata[0]["id"] == 1

    indexer.reset()
    assert indexer.faiss_index is None
    assert indexer.documents == []
    assert indexer.metadata == []


def test_index_without_metadata(dummy_data):
    embeddings, documents, _ = dummy_data
    indexer = DefaultIndexer()

    indexer.index(embeddings, documents)
    assert indexer.faiss_index.ntotal == 3
    assert indexer.metadata == [{}] * 3


def test_persist_creates_files(temp_index_dir, dummy_data):
    embeddings, documents, metadata = dummy_data
    indexer = DefaultIndexer(persist_path=temp_index_dir)

    indexer.index(embeddings, documents, metadata)
    indexer.persist()

    faiss_path = os.path.join(temp_index_dir, "index.faiss")
    pkl_path = os.path.join(temp_index_dir, "data.pkl")

    assert os.path.exists(faiss_path)
    assert os.path.exists(pkl_path)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
        assert "documents" in data
        assert "metadata" in data
        assert data["documents"] == documents
        assert data["metadata"] == metadata
