import pytest
from rag_src.chunker import DefaultChunker


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
    chunker = DefaultChunker(chunk_size=20, chunk_overlap=5)  # small to force chunking
    chunks = chunker.chunk(sample_docs, sample_metadata)

    assert len(chunks) > 0, "Chunks should not be empty"

    # Check metadata presence
    assert any("source: doc1" in chunk for chunk in chunks), "doc1 metadata not found"
    assert any("source: doc2" in chunk for chunk in chunks), "doc2 metadata not found"

    # Check some text is preserved
    assert any("first sentence" in chunk for chunk in chunks), "Expected text missing"


def test_chunking_without_metadata(sample_docs):
    chunker = DefaultChunker(chunk_size=20, chunk_overlap=5)
    chunks = chunker.chunk(sample_docs)

    assert len(chunks) > 0, "Chunks should not be empty"

    # Ensure no metadata is included
    assert all(
        "source:" not in chunk for chunk in chunks
    ), "Unexpected metadata in chunk"

    # Ensure content is there
    assert any("Another document begins here." in chunk for chunk in chunks)


def test_empty_input():
    chunker = DefaultChunker()
    chunks = chunker.chunk([])
    assert chunks == [], "Empty input should return empty list"


def test_chunk_count_behavior():
    # Make sure this creates multiple chunks
    long_text = "This is a sentence. " * 30  # ~600+ characters
    chunker = DefaultChunker(chunk_size=100, chunk_overlap=0)
    chunks = chunker.chunk([long_text])

    # Uncomment to debug
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i+1}:\n{chunk}\n")

    assert len(chunks) > 1, f"Expected >1 chunk but got {len(chunks)}"
