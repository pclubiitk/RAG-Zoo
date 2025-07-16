from rag_src.post_retrival_enricher import PostBaseEnricher
from typing import List
from llama_index.core.schema import Document
from llama_index.core.indices.document_summary import DocumentSummaryIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings as CoreSettings
from dotenv import load_dotenv
import os


class ContextualCompression(PostBaseEnricher):
    """
    Compress each document by summarizing them using Gemini LLM.
    """

    def __init__(self):
        """
        Initialize the ContextualCompression class by loading the environment and setting the global LLM.
        """
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment variables.")

        # Set global LLM via Settings
        CoreSettings.llm = GoogleGenAI(
            api_key=google_api_key,
            model="models/gemini-1.5-flash",
            temperature=0,
        )
        CoreSettings.embed_model = GoogleGenAIEmbedding(
            model_name="text-embedding-004", api_key=google_api_key
        )

    def enrich(self, docs: List[Document]) -> List[Document]:
        """
        Compress documents using the DocumentSummaryIndex with global Settings.

        Args:
            docs (List[Document]): A list of LlamaIndex Document objects.

        Returns:
            List[Document]: A list of summarized Document objects.
        """
        # Create a summary index from the input documents
        summary_index = DocumentSummaryIndex.from_documents(docs)

        # Generate and return summarized documents
        summarized_docs = []
        for doc in docs:
            summary = summary_index.get_document_summary(doc.doc_id)
            summarized_docs.append(Document(text=summary, doc_id=doc.doc_id))

        return summarized_docs
