import pytest
from rag_src.chunker import SemanticChunker
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


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
    chunker = SemanticChunker(chunk_size=30, chunk_overlap=5)
    chunks = chunker.chunk(sample_docs, sample_metadata)

    assert len(chunks) > 0, "Chunks should not be empty"
    assert any(
        "source: doc1" in chunk or "id: 001" in chunk for chunk in chunks
    ), "doc1 metadata missing"
    assert any(
        "source: doc2" in chunk or "id: 002" in chunk for chunk in chunks
    ), "doc2 metadata missing"


def test_chunking_without_metadata(sample_docs):
    chunker = SemanticChunker(chunk_size=30, chunk_overlap=5)
    chunks = chunker.chunk(sample_docs)

    assert len(chunks) > 0, "Chunks should not be empty"
    assert all(
        "source:" not in chunk and "id:" not in chunk for chunk in chunks
    ), "Unexpected metadata in chunk"


def test_empty_input():
    chunker = SemanticChunker()
    chunks = chunker.chunk([])
    assert chunks == [], "Expected empty list for empty input"


def test_chunk_count_behavior():
    long_text = "This is a semantic sentence. " * 40  # ~1000+ characters
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk([long_text])

    # Uncomment to debug
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i+1}: {chunk[:60]}...")

    assert len(chunks) > 1, f"Expected >1 chunk but got {len(chunks)}"
