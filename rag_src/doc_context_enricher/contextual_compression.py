from rag_src.doc_context_enricher.base import BaseContextEnricher
from typing import List
class contextual_compression(BaseContextEnricher):
  def __init__(self, llm):
    self.llm = llm
  def replacetabwithspace(documentslist):
    for doc in documentslist:
      doc.page_content=doc.page_content.replace('/t', ' ')
    return documentslist
  )
  def enrich(self, documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replacetabwithspace(texts)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)
    
    retriever = vector_store.as_retriever()
    
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
    
    qa_chain = RetrievalQA.from_chain_type(
      llm=llm,
      retriever=compression_retriever,
      return_source_documents=True
    )
    
    result = qa_chain.invoke({"query": query})
    return result["source documents"]
