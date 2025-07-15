from rag_src.postretrievaldoc_context_enricher.base import BasePostRetrievalContextEnricher
from typing import List

class ContextualCompression(BasePostRetrievalContextEnricher):
    """
    This module summarizes each document using the LLM
    It helps reduce the size and the noise in the retrieved context.
    """

    def __init__(self, llm):
        self.llm = llm

    def enrich(self, docs: List[str]) -> List[str]:
      import os
      import sys
      from dotenv import load_dotenv
      from langchain.retrievers.document_compressors import LLMChainExtractor
      from langchain.retrievers import ContextualCompressionRetriever
      from langchain.chains import RetrievalQA

      from google.colab import userdata

      GOOGLE_API_KEY=userdata.get("GOOGLE_API_KEY")

      def replacetabwithspace(documentslist):
        for doc in documentslist:
          doc.page_content=doc.page_content.replace('/t', ' ')
        return documentslist

      import os
      os.makedirs('data', exist_ok=True)

      from langchain_google_genai import GoogleGenerativeAIEmbeddings
      from langchain.document_loaders import PyPDFLoader
      from langchain.text_splitter import RecursiveCharacterTextSplitter
      from langchain_community.vectorstores import FAISS
      def encode_pdf(path, chunk_size=1000, chunk_overlap=200):

        loader = PyPDFLoader(path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        texts = text_splitter.split_documents(documents)
        cleaned_texts = replacetabwithspace(texts)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY) # Specify the Gemini embedding model
        vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

        return vectorstore

      vector_store = encode_pdf(path)

      from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
      from langchain.retrievers import ContextualCompressionRetriever
      from langchain.retrievers.document_compressors import LLMChainExtractor
      from langchain.chains import RetrievalQA

      retriever = vector_store.as_retriever()

      llm_compressor = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)
      compressor = LLMChainExtractor.from_llm(llm_compressor)

      compression_retriever = ContextualCompressionRetriever(
          base_compressor=compressor,
          base_retriever=retriever
      )

      llm_qa_chain = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)
      qa_chain = RetrievalQA.from_chain_type(
        llm=llm_qa_chain,
        retriever=compression_retriever,
        return_source_documents=True
      )

      query = "What is the main topic of the document?"
      result = qa_chain.invoke({"query": query})
      return(result["source_documents"]))
