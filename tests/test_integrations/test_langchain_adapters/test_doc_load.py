# tests/test_langchain_doc_loader_adapter.py

from langchain_core.documents import Document

from rag_src.doc_loader import DefaultDocLoader
from rag_src.integrations.langchain_adapters.doc_load import LangChainLoader


def test_loader_adapter(tmp_path):
    """
    Tests if the adapter correctly wraps a RAG-ZOO loader
    and outputs LangChain Document objects.
    """
    # 1. Setup: Create a dummy text file
    test_file = tmp_path / "test.txt"
    test_content = "This is a test document for RAG-ZOO."
    test_file.write_text(test_content)

    # 2. Initialize your RAG-ZOO loader and wrap it
    ragzoo_loader = DefaultDocLoader(path=str(test_file))
    adapter = LangChainLoader(ragzoo_loader=ragzoo_loader)

    # 3. Load documents using the adapter
    documents = adapter.load()

    # 4. Assert: Check if the output is correct
    assert isinstance(documents, list)
    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == test_content
    assert documents[0].metadata["source"] == str(test_file)
