from typing import Sequence
from langchain_core.documents import Document, BaseDocumentTransformer
from rag_src.doc_preprocessor.base import BasePreprocessor


class LangChainPreprocessor(BaseDocumentTransformer):
    """
    An adapter class that wraps any RAG-ZOO preprocessor and makes it
    compatible with the LangChain ecosystem as a Document Transformer.
    """

    def __init__(self, ragzoo_preprocessor: BasePreprocessor):
        """
        Initializes the adapter with an instance of a RAG-ZOO preprocessor.

        Args:
            ragzoo_preprocessor: An instance of a class that inherits from your
                                 BasePreprocessor.
        """
        self.ragzoo_preprocessor = ragzoo_preprocessor

    def transform_documents(self, documents: Sequence[Document], **kwargs) -> Sequence[Document]:
        """
        Transforms a sequence of Documents by applying the RAG-ZOO preprocessor
        to their page_content.

        Args:
            documents: A sequence of Documents to be transformed.

        Returns:
            A sequence of transformed Documents.
        """
        original_contents = [doc.page_content for doc in documents]
        cleaned_contents = self.ragzoo_preprocessor.preprocess(
            original_contents)
        transformed_documents = []
        for i, doc in enumerate(documents):
            new_doc = Document(
                page_content=cleaned_contents[i], metadata=doc.metadata)
            transformed_documents.append(new_doc)

        return transformed_documents

    async def atransform_documents(self, documents: Sequence[Document], **kwargs) -> Sequence[Document]:
        """Asynchronous version of the document transformation."""
        return self.transform_documents(documents, **kwargs)  # Since underlying preprocessor is synchronous, just calling the sync method.
