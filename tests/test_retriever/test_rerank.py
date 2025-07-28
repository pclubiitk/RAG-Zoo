import torch
import numpy as np
from unittest.mock import patch, MagicMock, mock_open


@patch("rag_src.retriever.default.pickle.load")
@patch("rag_src.retriever.default.faiss.read_index")
@patch("rag_src.retriever.default.os.path.exists", return_value=True)
@patch("rag_src.retriever.default.SentenceTransformer")
@patch("rag_src.retriever.rerank.AutoTokenizer.from_pretrained")
@patch("rag_src.retriever.rerank.AutoModelForSequenceClassification.from_pretrained")
@patch("rag_src.retriever.default.open", new_callable=mock_open)
def test_reranking_retriever(
    mock_open_file,
    mock_model_from_pretrained,
    mock_tokenizer_from_pretrained,
    mock_sentence_transformer,
    mock_os_exists,
    mock_faiss_read_index,
    mock_pickle_load,
):
    # Mock the SentenceTransformer encode output
    mock_sentence_transformer.return_value = MagicMock(
        encode=lambda x: np.array([[0.1, 0.2, 0.3]])
    )

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
    }
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer

    # Mock model logits
    mock_logits = MagicMock()
    mock_logits.view.return_value = torch.tensor([0.9, 0.1])
    mock_reranker_model = MagicMock()
    mock_reranker_model.eval = MagicMock()
    mock_reranker_model.return_value = MagicMock(logits=mock_logits)
    mock_model_from_pretrained.return_value = mock_reranker_model

    # Mock FAISS index with .search() return values
    mock_faiss_index = MagicMock()
    mock_faiss_index.search.return_value = (
        np.array([[0.1, 0.2]]),  # distances
        np.array([[0, 1]]),  # indices
    )
    mock_faiss_read_index.return_value = mock_faiss_index

    # Mock document store
    mock_pickle_load.return_value = {
        "documents": ["Document 1", "Document 2"],
        "metadata": [{"source": "A"}, {"source": "B"}],
    }

    # Instantiate retriever and test
    from rag_src.retriever.rerank import ReRankingRetriever

    retriever = ReRankingRetriever(index_path="dummy_index")
    results = retriever.retrieve("sample query", k=2)

    assert isinstance(results, list)
    assert len(results) == 2
