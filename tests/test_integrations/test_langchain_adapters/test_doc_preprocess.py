from langchain_core.documents import Document
from rag_src.doc_preprocessor import AdvancedPreprocessor
from rag_src.integrations.langchain_adapters.doc_preprocess import LangChainPreprocessor


def test_preprocessor_adapter():
    """
    Tests if the adapter correctly wraps the RAG-ZOO preprocessor
    and transforms the page_content of Document objects while preserving metadata.
    """
    # 1. Setup: Create sample documents with messy content and metadata
    docs_to_clean = [
        Document(
            page_content="<html><body>SOME  UPPERCASE text!</body></html>",
            metadata={"source": "doc_a", "id": 123}
        ),
        Document(
            page_content="Another document   with way too   many spaces.",
            metadata={"source": "doc_b", "id": 456}
        )
    ]

    # 2. Execution: Initialize the preprocessor and the adapter
    ragzoo_preprocessor = AdvancedPreprocessor()
    adapter = LangChainPreprocessor(ragzoo_preprocessor=ragzoo_preprocessor)

    # Transform the documents using the adapter
    transformed_docs = adapter.transform_documents(docs_to_clean)

    # 3. Assert: Check if the transformation was successful
    # Check that the list structure is correct
    assert isinstance(transformed_docs, list)
    assert len(transformed_docs) == 2

    # Check the first document
    assert transformed_docs[0].page_content == "some uppercase text!"
    assert transformed_docs[0].metadata == {"source": "doc_a", "id": 123}

    # Check the second document
    assert transformed_docs[1].page_content == "another document with way too many spaces."
    assert transformed_docs[1].metadata == {"source": "doc_b", "id": 456}
