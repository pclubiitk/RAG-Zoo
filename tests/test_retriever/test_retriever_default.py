import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from rag_src.retriever.default import DefaultRetriever

@pytest.fixture
def mock_faiss(monkeypatch):
    mock_index = MagicMock()
    mock_index.search.return_value = ([[0.1, 0.2, 0.3, 0.4, 0.5]], [[0, 1, 2, 3, 4]])
    monkeypatch.setattr("faiss.read_index", lambda x: mock_index)
    return mock_index

@pytest.fixture
def mock_pickle(monkeypatch):
    mock_data = {"documents": ["doc1", "doc2", "doc3", "doc4", "doc5"], "metadata": [{}, {}, {}, {}, {}]}
    monkeypatch.setattr("pickle.load", lambda f: mock_data)
    return mock_data

@pytest.fixture
def setup_files(tmp_path, monkeypatch, mock_faiss, mock_pickle):
    index_path = tmp_path / "index"
    index_path.mkdir()
    (index_path / "index.faiss").write_bytes(b"")
    (index_path / "data.pkl").write_bytes(b"")
    monkeypatch.setattr("os.path.exists", lambda x: True)
    return str(index_path)

@patch("rag_src.retriever.default.SentenceTransformer")
def test_default_retriever(mock_st, setup_files):
    mock_model = MagicMock()
    # Return a NumPy array, not a list
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_st.return_value = mock_model

    retriever = DefaultRetriever(index_path=setup_files)
    results = retriever.retrieve("query")

    assert isinstance(results, list)
    assert "text" in results[0]
    assert "metadata" in results[0]
