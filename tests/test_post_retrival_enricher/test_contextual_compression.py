from llama_index.core.schema import Document
from rag_src.post_retrival_enricher import ContextualCompression


def test_contextual_compression_basic():
    docs = [
        Document(
            text="LlamaIndex is a library that helps you connect LLMs to your data."
        ),
        Document(
            text="Gemini is a large language model by Google that supports summarization."
        ),
    ]

    enricher = ContextualCompression()
    result = enricher.enrich(docs)

    assert isinstance(result, list)
    assert all(isinstance(doc, Document) for doc in result)
    assert all(doc.text.strip() != "" for doc in result)
