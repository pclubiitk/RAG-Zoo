import pytest
from rag_src.chunker import RecursiveChunker


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
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk(sample_docs, sample_metadata)

    assert len(chunks) > 0, "Chunks should not be empty"

    # Metadata should be embedded by LlamaIndex
    assert any(
        "doc1" in chunk or "001" in chunk for chunk in chunks
    ), "doc1 metadata missing"
    assert any(
        "doc2" in chunk or "002" in chunk for chunk in chunks
    ), "doc2 metadata missing"


def test_chunking_without_metadata(sample_docs):
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk(sample_docs)

    assert len(chunks) > 0, "Chunks should not be empty"

    # Should not raise error without metadata
    assert all(isinstance(chunk, str) for chunk in chunks), "Chunks must be strings"


def test_empty_input():
    chunker = RecursiveChunker()
    chunks = chunker.chunk([])
    assert chunks == [], "Expected empty list on empty input"


def test_chunk_count_behavior():
    long_text = "This is a sentence. " * 50  # ~1000+ characters
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
    chunks = chunker.chunk([long_text])

    # Uncomment to inspect chunks
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i+1}: {chunk[:60]}...")

    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
