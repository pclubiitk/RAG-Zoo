import pytest
from rag_src.chunker import TokenChunker


@pytest.fixture
def sample_docs():
    return [
        "This is the first sentence. Here is the second sentence. And the third one.",
        "Another document begins here. It continues with more text. Ending now.",
    ]


@pytest.fixture
def sample_metadata():
    return [{"source": "doc1", "id": "001"}, {"source": "doc2", "id": "002"}]


def test_chunking_with_metadata(sample_docs, sample_metadata):
    chunker = TokenChunker(chunk_size=20, chunk_overlap=5)
    chunks = chunker.chunk(sample_docs, sample_metadata)

    assert len(chunks) > 0, "Chunks should not be empty"

    # Metadata should be present in chunks
    assert any(
        "source: doc1" in chunk or "id: 001" in chunk for chunk in chunks
    ), "Metadata from doc1 missing"
    assert any(
        "source: doc2" in chunk or "id: 002" in chunk for chunk in chunks
    ), "Metadata from doc2 missing"


def test_chunking_without_metadata(sample_docs):
    chunker = TokenChunker(chunk_size=20, chunk_overlap=5)
    chunks = chunker.chunk(sample_docs)

    assert len(chunks) > 0, "Chunks should not be empty"
    assert all(
        "source:" not in chunk and "id:" not in chunk for chunk in chunks
    ), "Unexpected metadata found"


def test_empty_input():
    chunker = TokenChunker()
    chunks = chunker.chunk([])
    assert chunks == [], "Expected empty list for empty input"


def test_chunk_count_behavior():
    long_text = "This sentence repeats. " * 50  # ~1000 tokens worth
    chunker = TokenChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk([long_text])

    assert len(chunks) > 1, "Expected multiple chunks for long input"
